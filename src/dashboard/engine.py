"""
AIravat engine — SIH26078 (MoES)
AI-driven spatio-temporal tracking of extreme weather anomalies in medium-range forecasts.

Everything the dashboard shows is computed here:
  * exact closed-form discrete Lalaurette EFI + Shift of Tails (SOT)
  * scenario generator (22-member ensemble + 30-yr climatology) used for the demo
  * Haversine DBSCAN clustering -> 640 x 640 km bounding boxes
  * cyclone track + uncertainty cone
  * downscaling concept demo (12 km -> 5 km) with radially averaged PSD
  * divergence-free projection (mass-continuity check)
  * 4-tier severity scoring

The demo data is SYNTHETIC. Swap `build_scenario()` for the real NEPS-G / ERA5 loaders
and `generative_sample()` for the trained diffusion model; every other function keeps working.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter, zoom

EARTH_R_KM = 6371.0
N_MEMBERS = 23
N_CLIM = 930  # 30 years x 31-day window
LEADS = list(range(72, 241, 24))

LAT = np.arange(5.0, 37.01, 0.5)
LON = np.arange(66.0, 98.01, 0.5)
LON2D, LAT2D = np.meshgrid(LON, LAT)

TIERS = [
    # name, colour, one-line protocol
    ("NORMAL", "#2E9E5B", "No action. Routine monitoring continues."),
    ("ADVISORY", "#E8C547", "Advisory bulletin to district agriculture and irrigation departments."),
    ("WATCH", "#EE8A2F", "Activate DEOC, cancel leave in vulnerable blocks, inspect shelters."),
    ("WARNING", "#D6453D", "Pre-position SDRF/NDRF, order evacuation of vulnerable coastal and riverine zones."),
    ("EMERGENCY", "#8E1B4A", "Full mobilisation: complete evacuation, marine and travel suspension, civil defence."),
]

SCENARIOS = {
    "Bay of Bengal cyclone (post-monsoon)": {
        "key": "cyclone", "variable": "precip", "unit": "mm/day", "cycle": "2026-10-20 00 UTC",
        "thresholds": [64.5, 115.5, 204.4], "hazard": "Tropical cyclone: extreme rain and gale",
    },
    "Arabian Sea severe cyclone (Saurashtra / Kutch)": {
        "key": "cyclone_arabian", "variable": "precip", "unit": "mm/day", "cycle": "2026-06-12 00 UTC",
        "thresholds": [64.5, 115.5, 204.4], "hazard": "Severe cyclonic storm: coastal surge and deluge",
    },
    "Himalayan cloudburst & flash flood (Uttarakhand)": {
        "key": "cloudburst", "variable": "precip", "unit": "mm/day", "cycle": "2026-08-08 00 UTC",
        "thresholds": [64.5, 115.5, 204.4], "hazard": "Mountain convective cloudburst & flash flood",
    },
    "Brahmaputra basin extreme deluge (Assam)": {
        "key": "assam_flood", "variable": "precip", "unit": "mm/day", "cycle": "2026-06-28 00 UTC",
        "thresholds": [64.5, 115.5, 204.4], "hazard": "Prolonged monsoon depression & basin flooding",
    },
    "Northwest India severe heatwave (pre-monsoon)": {
        "key": "heat", "variable": "t2m", "unit": "°C", "cycle": "2026-05-18 00 UTC",
        "thresholds": [42.0, 45.0, 47.0], "hazard": "Heat dome: severe heatwave",
    },
    "Monsoon depression + Konkan heavy rain": {
        "key": "monsoon", "variable": "precip", "unit": "mm/day", "cycle": "2026-07-14 00 UTC",
        "thresholds": [64.5, 115.5, 204.4], "hazard": "Monsoon low / orographic heavy rain",
    },
    "Coromandel coastal deluge / NE monsoon (Chennai)": {
        "key": "chennai_flood", "variable": "precip", "unit": "mm/day", "cycle": "2026-11-26 00 UTC",
        "thresholds": [64.5, 115.5, 204.4], "hazard": "Northeast monsoon coastal cloudburst deluge",
    },
}

PLACES = {
    "Puri, Odisha": (19.81, 85.83),
    "Bhubaneswar, Odisha": (20.30, 85.82),
    "Visakhapatnam, Andhra Pradesh": (17.69, 83.22),
    "Kolkata, West Bengal": (22.57, 88.36),
    "Dehradun, Uttarakhand": (30.32, 78.03),
    "Shimla, Himachal Pradesh": (31.10, 77.17),
    "Rishikesh, Uttarakhand": (30.08, 78.27),
    "Guwahati, Assam": (26.14, 91.74),
    "Shillong, Meghalaya": (25.57, 91.88),
    "Dibrugarh, Assam": (27.47, 94.91),
    "Porbandar, Gujarat": (21.64, 69.60),
    "Dwarka, Gujarat": (22.24, 68.96),
    "Bhuj, Gujarat": (23.24, 69.66),
    "Ahmedabad, Gujarat": (23.02, 72.57),
    "Jaisalmer, Rajasthan": (26.92, 70.91),
    "Bikaner, Rajasthan": (28.02, 73.31),
    "Jaipur, Rajasthan": (26.91, 75.79),
    "New Delhi": (28.61, 77.21),
    "Ratnagiri, Maharashtra": (16.99, 73.31),
    "Mumbai, Maharashtra": (19.08, 72.88),
    "Pune, Maharashtra": (18.52, 73.86),
    "Nagpur, Maharashtra": (21.15, 79.09),
    "Bhopal, Madhya Pradesh": (23.26, 77.41),
    "Raipur, Chhattisgarh": (21.25, 81.63),
    "Hyderabad, Telangana": (17.39, 78.49),
    "Chennai, Tamil Nadu": (13.08, 80.27),
    "Cuddalore, Tamil Nadu": (11.75, 79.77),
    "Bengaluru, Karnataka": (12.97, 77.59),
    "Kochi, Kerala": (9.93, 76.27),
    "Srinagar, Jammu & Kashmir": (34.08, 74.80),
    "Patna, Bihar": (25.59, 85.14),
    "Lucknow, Uttar Pradesh": (26.85, 80.95),
}


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def km_to_deg_lat(km):
    return km / 111.2


def km_to_deg_lon(km, lat):
    return km / (111.2 * np.cos(np.radians(lat)))


def circle(lat, lon, radius_km, n=48):
    t = np.linspace(0, 2 * np.pi, n)
    return lat + km_to_deg_lat(radius_km) * np.sin(t), lon + km_to_deg_lon(radius_km, lat) * np.cos(t)


# ----------------------------------------------------------------------------------------
# Extreme Forecast Index (exact closed form) and Shift of Tails
# ----------------------------------------------------------------------------------------
def _psi(p, k):
    """Antiderivative of (p - k)/sqrt(p(1-p)):  -sqrt(p(1-p)) + (1/2 - k) arcsin(2p - 1)."""
    p = np.clip(p, 0.0, 1.0)
    return -np.sqrt(np.maximum(0.0, p * (1.0 - p))) + (0.5 - k) * np.arcsin(np.clip(2 * p - 1, -1, 1))


def efi_from_probs(p_sorted: np.ndarray) -> np.ndarray:
    """
    Vectorised exact EFI.
    p_sorted: (..., M) climatological CDF values F_c(x_(m)) of the SORTED ensemble members.
    Returns EFI in [-1, 1] with shape (...).
    """
    m_count = p_sorted.shape[-1]
    zeros = np.zeros(p_sorted.shape[:-1] + (1,))
    p = np.concatenate([zeros, np.clip(p_sorted, 0, 1), zeros + 1.0], axis=-1)
    k = np.arange(m_count + 1) / m_count
    terms = _psi(p[..., 1:], k) - _psi(p[..., :-1], k)
    return np.clip((2.0 / np.pi) * terms.sum(axis=-1), -1.0, 1.0)


def efi_discrete(ensemble: np.ndarray, climatology: np.ndarray) -> float:
    """Exact EFI for one point from raw ensemble members and raw climatology samples (ECDF)."""
    ens = np.sort(np.asarray(ensemble, dtype=float))
    clim = np.sort(np.asarray(climatology, dtype=float))
    p_m = np.searchsorted(clim, ens, side="right") / clim.size
    return float(efi_from_probs(p_m))


def sot(q_f90, q_c90, q_c99):
    """Shift of Tails (upper tail): (Qf(0.90) - Qc(0.99)) / (Qc(0.99) - Qc(0.90))."""
    denom = np.maximum(np.asarray(q_c99) - np.asarray(q_c90), 1e-6)
    return (np.asarray(q_f90) - np.asarray(q_c99)) / denom


# ----------------------------------------------------------------------------------------
# Synthetic scenario generator (stands in for NEPS-G members + ERA5 climatology)
# ----------------------------------------------------------------------------------------
def _gauss_blob(lat0, lon0, radius_km, lat=LAT2D, lon=LON2D):
    d = haversine_km(lat0, lon0, lat, lon)
    return np.exp(-0.5 * (d / radius_km) ** 2)


def cyclone_track(lead_h, key="cyclone"):
    """Ensemble-mean centre of the cyclone."""
    if "arabian" in str(key):
        # Arabian Sea track: starts in central sea, heads NNE to Saurashtra / Kutch (Porbandar/Dwarka)
        waypoints = np.array([
            [72, 16.5, 67.2], [96, 18.2, 67.8], [120, 20.1, 68.6], [144, 21.8, 69.4],
            [168, 23.2, 70.1], [192, 24.1, 70.8], [216, 24.8, 71.5], [240, 25.4, 72.0],
        ])
    else:
        # Bay of Bengal track: starts in central bay, heads NW to Odisha (Puri/Gopalpur)
        waypoints = np.array([
            [72, 15.2, 88.4], [96, 16.9, 87.3], [120, 18.8, 86.3], [144, 20.4, 85.4],
            [168, 21.6, 84.4], [192, 22.5, 83.3], [216, 23.1, 82.2], [240, 23.6, 81.2],
        ])
    lat = np.interp(lead_h, waypoints[:, 0], waypoints[:, 1])
    lon = np.interp(lead_h, waypoints[:, 0], waypoints[:, 2])
    return float(lat), float(lon)


def track_cone(lead_h):
    """Cone radius (km) grows with lead time: ~45 km at T+72 h, ~180 km at T+240 h."""
    return 45.0 + 0.8 * (lead_h - 72)


def cyclone_mslp(lead_h):
    # deepens until landfall, fills afterwards
    return float(np.interp(lead_h, [72, 120, 132, 168, 240], [992, 968, 962, 984, 998]))


def _scenario_centres(key, lead_h):
    if key == "cyclone":
        lat, lon = cyclone_track(lead_h, "cyclone")
        land = lead_h > 128
        amp = 260.0 * (0.55 if land else 1.0) * np.interp(lead_h, [72, 120, 240], [0.7, 1.0, 0.6])
        return [(lat, lon, 150.0, amp)]
    if key == "cyclone_arabian":
        lat, lon = cyclone_track(lead_h, "cyclone_arabian")
        land = lead_h > 140
        amp = 250.0 * (0.60 if land else 1.0) * np.interp(lead_h, [72, 144, 240], [0.75, 1.0, 0.55])
        return [(lat, lon, 140.0, amp)]
    if key == "cloudburst":
        # Intense high-altitude convective anomaly over Garhwal / Kumaon Himalaya
        lat = 30.4 + 0.005 * (lead_h - 72)
        lon = 78.4 + 0.008 * (lead_h - 72)
        amp = float(np.interp(lead_h, [72, 120, 144, 240], [140, 260, 220, 90]))
        return [(lat, lon, 90.0, amp)]
    if key == "assam_flood":
        # Broad torrential monsoon deluge over Brahmaputra valley
        lat = 26.2 + 0.003 * (lead_h - 72)
        lon = 91.8 + 0.005 * (lead_h - 72)
        amp = float(np.interp(lead_h, [72, 120, 168, 240], [160, 230, 200, 140]))
        return [(lat, lon, 170.0, amp)]
    if key == "chennai_flood":
        # Coastal convergence cloudburst over Tamil Nadu coast
        lat = 13.1 + 0.006 * (lead_h - 72)
        lon = 80.3 - 0.004 * (lead_h - 72)
        amp = float(np.interp(lead_h, [72, 120, 168, 240], [150, 240, 190, 95]))
        return [(lat, lon, 120.0, amp)]
    if key == "heat":
        lat = 27.2 + 0.006 * (lead_h - 72)
        lon = 72.8 + 0.018 * (lead_h - 72)
        amp = float(np.interp(lead_h, [72, 144, 240], [5.5, 8.2, 6.0]))
        return [(lat, lon, 330.0, amp)]
    # monsoon depression moving WNW + west coast orographic band
    lat = 21.8 + 0.010 * (lead_h - 72)
    lon = 84.5 - 0.045 * (lead_h - 72)
    amp = float(np.interp(lead_h, [72, 144, 240], [140, 190, 120]))
    return [(lat, lon, 180.0, amp)]


def _konkan_band():
    band = np.exp(-0.5 * ((LON2D - 73.7) / 0.45) ** 2) * ((LAT2D > 14.0) & (LAT2D < 20.5))
    return gaussian_filter(band.astype(float), 1.0)


def climatology_params(variable, key):
    """Parameters of the 30-year climatology at every grid point."""
    if variable == "precip":
        base = 5.0 + 14.0 * np.exp(-0.5 * ((LON2D - 73.8) / 0.8) ** 2) * ((LAT2D > 10) & (LAT2D < 21))
        base += 12.0 * _gauss_blob(25.5, 91.5, 250)  # north-east India
        base += 4.0 * _gauss_blob(20.0, 86.0, 400) if key == "cyclone" else 0.0
        shape = 0.55
        return {"dist": "gamma", "shape": shape, "scale": base / shape}
    mean = 38.5 - 0.35 * np.abs(LAT2D - 26.5) - 3.0 * (LON2D > 88)
    return {"dist": "norm", "loc": mean, "scale": np.full_like(mean, 2.1)}


def _clim_cdf(x, cp):
    if cp["dist"] == "gamma":
        return stats.gamma.cdf(x, cp["shape"], scale=cp["scale"][..., None])
    return stats.norm.cdf(x, loc=cp["loc"][..., None], scale=cp["scale"][..., None])


def _clim_ppf(q, cp):
    if cp["dist"] == "gamma":
        return stats.gamma.ppf(q, cp["shape"], scale=cp["scale"])
    return stats.norm.ppf(q, loc=cp["loc"], scale=cp["scale"])


def climatology_samples(cp, i, j, rng):
    """930 climatological samples at one grid point (used by the exact ECDF EFI)."""
    if cp["dist"] == "gamma":
        return rng.gamma(cp["shape"], cp["scale"][i, j], N_CLIM)
    return rng.normal(cp["loc"][i, j], cp["scale"][i, j], N_CLIM)


def build_scenario(scenario_name: str, lead_h: int, seed: int = 7):
    """
    Returns a dict with the 22-member ensemble (M, ny, nx), EFI, SOT and exceedance probability.
    """
    cfg = SCENARIOS[scenario_name]
    key, var = cfg["key"], cfg["variable"]
    rng = np.random.default_rng(seed + lead_h)
    cp = climatology_params(var, key)
    spread_km = 25.0 + 0.9 * (lead_h - 72)  # ensemble position spread grows with lead

    members = []
    for _ in range(N_MEMBERS):
        if var == "precip":
            field = rng.gamma(cp["shape"], cp["scale"]) * 0.6 + cp["scale"] * cp["shape"] * 0.4
        else:
            field = cp["loc"] + rng.normal(0, 0.6, LAT2D.shape)
        for lat0, lon0, rad, amp in _scenario_centres(key, lead_h):
            dlat = km_to_deg_lat(rng.normal(0, spread_km))
            dlon = km_to_deg_lon(rng.normal(0, spread_km), lat0)
            field = field + amp * rng.uniform(0.75, 1.2) * _gauss_blob(lat0 + dlat, lon0 + dlon, rad)
        if key == "monsoon":
            field = field + rng.uniform(110, 190) * _konkan_band()
        members.append(field)
    ens = np.stack(members)  # (M, ny, nx)

    ens_last = np.moveaxis(ens, 0, -1)  # (ny, nx, M)
    p_sorted = _clim_cdf(np.sort(ens_last, axis=-1), cp)
    efi = efi_from_probs(p_sorted)
    q_f90 = np.percentile(ens_last, 90, axis=-1)
    s = sot(q_f90, _clim_ppf(0.90, cp), _clim_ppf(0.99, cp))
    crit = cfg["thresholds"][1]
    p_exc = (ens_last >= crit).mean(axis=-1)
    return {"cfg": cfg, "ens": ens, "efi": efi, "sot": s, "p_exc": p_exc, "clim": cp, "lead": lead_h}


# ----------------------------------------------------------------------------------------
# Clustering -> 4D bounding boxes
# ----------------------------------------------------------------------------------------
def cluster_anomalies(sc, efi_thresh=0.65, eps_km=200.0, min_samples=8):
    from sklearn.cluster import DBSCAN

    efi, s = sc["efi"], sc["sot"]
    mask = (np.abs(efi) >= efi_thresh) & (s >= 0.0)
    if mask.sum() < min_samples:
        return []
    lats, lons = LAT2D[mask], LON2D[mask]
    coords = np.radians(np.column_stack([lats, lons]))
    labels = DBSCAN(eps=eps_km / EARTH_R_KM, min_samples=min_samples, metric="haversine").fit_predict(coords)
    e_vals, s_vals, p_vals = efi[mask], s[mask], sc["p_exc"][mask]
    ens_max = sc["ens"].max(axis=0)[mask]
    out = []
    for lab in sorted(set(labels) - {-1}):
        idx = labels == lab
        w = np.abs(e_vals[idx]) * (1 + np.maximum(s_vals[idx], 0))
        la, lo = np.radians(lats[idx]), np.radians(lons[idx])
        v = np.array([(w * np.cos(la) * np.cos(lo)).sum(), (w * np.cos(la) * np.sin(lo)).sum(), (w * np.sin(la)).sum()])
        v /= np.linalg.norm(v)
        c_lat, c_lon = float(np.degrees(np.arcsin(v[2]))), float(np.degrees(np.arctan2(v[1], v[0])))
        half = 320.0  # 128 x 128 cells at 5 km = 640 km box
        out.append({
            "id": f"ANOM-{sc['cfg']['key'].upper()[:3]}-{lab + 1:02d}",
            "nodes": int(idx.sum()),
            "centroid": (round(c_lat, 2), round(c_lon, 2)),
            "box": (c_lat - km_to_deg_lat(half), c_lat + km_to_deg_lat(half),
                    c_lon - km_to_deg_lon(half, c_lat), c_lon + km_to_deg_lon(half, c_lat)),
            "max_efi": float(e_vals[idx].max()),
            "max_sot": float(s_vals[idx].max()),
            "max_prob": float(p_vals[idx].max()),
            "peak_value": float(ens_max[idx].max()),
        })
    return sorted(out, key=lambda c: -c["max_efi"])


# ----------------------------------------------------------------------------------------
# Severity engine
# ----------------------------------------------------------------------------------------
def severity(efi, sot_val, prob):
    score = 0.35 * max(efi, 0) + 0.25 * min(1.0, max(sot_val, 0) / 2.0) + 0.40 * prob
    if efi >= 0.95 and sot_val > 2.0 and prob >= 0.85:
        tier = 4
    elif efi >= 0.85 and sot_val > 1.0 and prob >= 0.65:
        tier = 3
    elif efi >= 0.70 and sot_val > 0 and prob >= 0.30:
        tier = 2
    elif efi >= 0.50:
        tier = 1
    else:
        tier = 0
    return score, tier


def point_diagnostics(sc, lat, lon, seed=11):
    i = int(np.abs(LAT - lat).argmin())
    j = int(np.abs(LON - lon).argmin())
    rng = np.random.default_rng(seed + i * 1000 + j)
    ens = sc["ens"][:, i, j]
    clim = climatology_samples(sc["clim"], i, j, rng)
    efi_val = efi_discrete(ens, clim)
    sot_val = float(sot(np.percentile(ens, 90), np.percentile(clim, 90), np.percentile(clim, 99)))
    thr = sc["cfg"]["thresholds"]
    probs = {t: float((ens >= t).mean()) for t in thr}
    score, tier = severity(efi_val, sot_val, probs[thr[1]])
    return {
        "grid_lat": float(LAT[i]), "grid_lon": float(LON[j]),
        "distance_km": float(haversine_km(lat, lon, LAT[i], LON[j])),
        "efi": efi_val, "sot": sot_val, "probs": probs, "score": score, "tier": tier,
        "q50": float(np.percentile(ens, 50)), "q90": float(np.percentile(ens, 90)), "q99": float(np.percentile(ens, 99)),
        "clim_q99": float(np.percentile(clim, 99)), "clim_q90": float(np.percentile(clim, 90)), "members": ens, "clim_samples": clim,
    }


# ----------------------------------------------------------------------------------------
# Downscaling concept demo (12 km -> 5 km, 128 x 128)
# ----------------------------------------------------------------------------------------
def _spectral_field(n, rng, slope=-5.0 / 3.0):
    """Gaussian random field whose isotropic energy spectrum follows E(k) ~ k^slope."""
    kx = np.fft.fftfreq(n)
    kk = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
    kk[0, 0] = 1.0
    amp = kk ** ((slope - 1.0) / 2.0)  # |u_k|^2 * k ~ E(k)
    amp[0, 0] = 0.0
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    f = np.real(np.fft.ifft2(amp * phase))
    return (f - f.mean()) / f.std(), kk


def _split_scales(f, kk, k_cut):
    F = np.fft.fft2(f)
    low = np.real(np.fft.ifft2(F * (kk <= k_cut)))
    return low, f - low


def radial_psd(field, dx_km=5.0):
    n = field.shape[0]
    F = np.fft.fftshift(np.abs(np.fft.fft2(field - field.mean())) ** 2)
    y, x = np.indices(F.shape)
    r = np.hypot(x - n // 2, y - n // 2).astype(int)
    psd = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = np.arange(psd.size) / (n * dx_km)  # cycles per km
    sel = slice(1, n // 2)
    return k[sel], psd[sel]


def _to_physical(z, variable, peak_hint):
    if variable == "precip":
        scale = peak_hint / np.exp(2.9)
        return np.maximum(scale * np.exp(0.95 * z) - 0.3 * scale, 0.0)
    return peak_hint - 4.0 + 1.6 * z


def generative_sample(z_low, kk, k_cut, rng):
    """
    Stand-in for the conditional diffusion sampler: keeps the resolved (12 km) scales and draws
    NEW sub-grid detail with the correct k^-5/3 spectrum. Replace with the trained EDM/DDPM call.
    """
    z_new, _ = _spectral_field(z_low.shape[0], rng)
    _, high = _split_scales(z_new, kk, k_cut)
    return z_low + high


def downscale_demo(variable="precip", peak_hint=180.0, seed=3, n_real=4):
    rng = np.random.default_rng(seed)
    n = 128
    z, kk = _spectral_field(n, rng)
    k_cut = 5.0 / 24.0  # 12 km grid resolves wavelengths >= 24 km (Nyquist) -> k in cycles per 5-km pixel
    z_low, _ = _split_scales(z, kk, k_cut)
    truth = _to_physical(z, variable, peak_hint)
    coarse_small = zoom(gaussian_filter(truth, 1.2), 5.0 / 12.0, order=1)
    coarse = zoom(coarse_small, n / coarse_small.shape[0], order=0)[:n, :n]
    mse = gaussian_filter(truth, 2.4)  # conditional-mean behaviour of an MSE-trained network
    gens = [_to_physical(generative_sample(z_low, kk, k_cut, rng), variable, peak_hint) for _ in range(n_real)]
    return {"truth": truth, "coarse": coarse, "mse": mse, "gens": gens}


# ----------------------------------------------------------------------------------------
# Mass-continuity check with test-time projection (Helmholtz, periodic box)
# ----------------------------------------------------------------------------------------
def divergence(u, v, dx_m=5000.0):
    return (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx_m) + (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dx_m)


def project_divergence_free(u, v, dx_m=5000.0):
    """Removes the divergent part of (u, v) consistently with the centred-difference operator."""
    n = u.shape[0]
    k = np.fft.fftfreq(n)
    sx = np.sin(2 * np.pi * k)[None, :] / dx_m  # spectral symbol of the centred difference
    sy = np.sin(2 * np.pi * k)[:, None] / dx_m
    U, V = np.fft.fft2(u), np.fft.fft2(v)
    d = sx * U + sy * V
    s2 = sx ** 2 + sy ** 2
    s2[s2 == 0] = 1.0
    U2, V2 = U - sx * d / s2, V - sy * d / s2
    return np.real(np.fft.ifft2(U2)), np.real(np.fft.ifft2(V2))


def wind_demo(seed=5):
    rng = np.random.default_rng(seed)
    psi, _ = _spectral_field(128, rng, slope=-4.0)
    chi, _ = _spectral_field(128, rng, slope=-4.0)
    chi = gaussian_filter(chi, 2.0, mode="wrap")
    # rotational flow from a streamfunction + a spurious divergent part (what an unconstrained generator adds)
    def d(f, ax):  # periodic centred difference (the box is treated as periodic)
        return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / 2.0

    u_rot, v_rot = -d(psi, 0), d(psi, 1)
    u_div, v_div = d(chi, 1), d(chi, 0)
    scale_rot = 12.0 / np.sqrt((u_rot ** 2 + v_rot ** 2).mean())
    scale_div = 1.5 / np.sqrt((u_div ** 2 + v_div ** 2).mean())
    u = u_rot * scale_rot + u_div * scale_div
    v = v_rot * scale_rot + v_div * scale_div
    up, vp = project_divergence_free(u, v)
    return {"u": u, "v": v, "up": up, "vp": vp, "div": divergence(u, v), "divp": divergence(up, vp)}
