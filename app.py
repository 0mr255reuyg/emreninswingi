"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          BIST 500 - SWING TRADE TARAMA VE PUANLAMA SİSTEMİ                 ║
║          app.py - Streamlit Arayüzü + Tarama Motoru                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# BIST HİSSE LİSTESİ
# ─────────────────────────────────────────────────────────────────────────────

BIST_LISTESI = [
    "AKBNK.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS", "VAKBN.IS", "YKBNK.IS",
    "QNBFB.IS", "TSKB.IS", "ALBRK.IS", "KLNMA.IS",
    "KCHOL.IS", "SAHOL.IS", "SISE.IS", "KOZAA.IS", "KOZAL.IS", "TUPRS.IS",
    "EREGL.IS", "ARCLK.IS", "BIMAS.IS", "MIGROS.IS", "TCELL.IS",
    "THYAO.IS", "PGSUS.IS", "ULKER.IS", "AEFES.IS",
    "ENKAI.IS", "AYGAZ.IS", "DOHOL.IS", "PETKM.IS", "GUBRF.IS", "EKGYO.IS",
    "TOASO.IS", "FROTO.IS", "OTKAR.IS", "TTRAK.IS",
    "ASELS.IS", "LOGO.IS", "NETAS.IS", "KAREL.IS", "ARENA.IS",
    "ISGYO.IS", "TRGYO.IS", "ALGYO.IS",
    "ECILC.IS", "DEVA.IS", "ECZYT.IS",
    "KRDMD.IS", "CIMSA.IS", "AKCNS.IS", "BOLUC.IS",
    "TTKOM.IS", "VESBE.IS", "BRISA.IS",
    "TRKCM.IS", "SODA.IS", "BAGFS.IS",
    "HEKTS.IS", "BIZIM.IS", "TAVHL.IS",
    "AKSEN.IS", "ZOREN.IS", "CLEBI.IS",
    "AGESA.IS", "AKSA.IS", "SOKM.IS", "MAVI.IS",
]
BIST_LISTESI = list(set(BIST_LISTESI))


# ─────────────────────────────────────────────────────────────────────────────
# GÖSTERGE HESAPLAMA FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────

def hesapla_rsi(fiyatlar: pd.Series, periyot: int = 14) -> float:
    delta = fiyatlar.diff()
    kazan = delta.clip(lower=0)
    kayip = -delta.clip(upper=0)
    ort_kazan = kazan.ewm(com=periyot - 1, adjust=False).mean()
    ort_kayip = kayip.ewm(com=periyot - 1, adjust=False).mean()
    rs = ort_kazan / ort_kayip
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def hesapla_macd(fiyatlar: pd.Series, hizli=12, yavas=26, sinyal=9):
    ema_hizli = fiyatlar.ewm(span=hizli, adjust=False).mean()
    ema_yavas = fiyatlar.ewm(span=yavas, adjust=False).mean()
    macd_serisi = ema_hizli - ema_yavas
    sinyal_serisi = macd_serisi.ewm(span=sinyal, adjust=False).mean()
    histogram = macd_serisi - sinyal_serisi
    return (
        macd_serisi.iloc[-1],
        sinyal_serisi.iloc[-1],
        histogram.iloc[-1],
        histogram.iloc[-2] if len(histogram) > 1 else 0
    )


def hesapla_atr(df: pd.DataFrame, periyot: int = 14) -> float:
    yuksek = df['High']
    dusuk = df['Low']
    kapanis = df['Close']
    tr1 = yuksek - dusuk
    tr2 = abs(yuksek - kapanis.shift())
    tr3 = abs(dusuk - kapanis.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=periyot, adjust=False).mean()
    return atr.iloc[-1]


def hesapla_ma(fiyatlar: pd.Series, periyot: int) -> float:
    if len(fiyatlar) < periyot:
        return np.nan
    return fiyatlar.rolling(window=periyot).mean().iloc[-1]


# ─────────────────────────────────────────────────────────────────────────────
# PUANLAMA FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────────────────

def puan_pddd(pddd):
    if pddd is None or (isinstance(pddd, float) and np.isnan(pddd)) or pddd <= 0:
        return 0, "Veri yok"
    ref = 2.0
    if pddd < 1.0:   return 15, f"Çok Ucuz ({pddd:.2f})"
    elif pddd < 1.5: return 12, f"Ucuz ({pddd:.2f})"
    elif pddd < ref: return 8,  f"Makul ({pddd:.2f})"
    elif pddd < ref * 3: return 3, f"Pahalı ({pddd:.2f})"
    else:            return 0,  f"Çok Pahalı ({pddd:.2f})"


def puan_fk(fk):
    if fk is None or (isinstance(fk, float) and np.isnan(fk)) or fk <= 0:
        return 0, "Zarar / Veri yok"
    ref = 18.0
    if fk < 10:      return 15, f"Çok Ucuz ({fk:.1f}x)"
    elif fk < 15:    return 12, f"Ucuz ({fk:.1f}x)"
    elif fk < ref:   return 8,  f"Makul ({fk:.1f}x)"
    elif fk < ref*2: return 3,  f"Pahalı ({fk:.1f}x)"
    else:            return 0,  f"Çok Pahalı ({fk:.1f}x)"


def puan_kar_buyumesi(buyume):
    if buyume is None or (isinstance(buyume, float) and np.isnan(buyume)):
        return 3, "Veri yok"
    if buyume > 50:  return 10, f"Güçlü Büyüme (%{buyume:.0f})"
    elif buyume > 20: return 8, f"İyi Büyüme (%{buyume:.0f})"
    elif buyume > 0:  return 5, f"Zayıf Büyüme (%{buyume:.0f})"
    else:             return 0, f"Küçülme (%{buyume:.0f})"


def puan_trend(fiyat, ma50, ma200):
    if np.isnan(ma50) or np.isnan(ma200):
        return 0, "MA verisi yok", False
    f_ma50  = fiyat > ma50
    f_ma200 = fiyat > ma200
    ma50_ma200 = ma50 > ma200
    if f_ma50 and f_ma200 and ma50_ma200:
        return 15, "Güçlü Trend ↑ (Golden)", True
    elif f_ma50 and f_ma200:
        return 10, "Pozitif Trend ↑", True
    elif f_ma200 and not f_ma50:
        return 5, "Zayıf / Konsolidasyon", True
    else:
        return 0, "Düşüş Trendi ↓ (ELENDİ)", False


def puan_rsi(rsi):
    if np.isnan(rsi): return 5, "Veri yok"
    if rsi < 30:      return 3,  f"Aşırı Satım ({rsi:.1f})"
    elif rsi < 50:    return 7,  f"Nötr ({rsi:.1f})"
    elif rsi < 65:    return 15, f"İdeal Bölge ✓ ({rsi:.1f})"
    elif rsi < 70:    return 10, f"Güçlü ({rsi:.1f})"
    elif rsi < 80:    return 3,  f"Aşırı Alım ({rsi:.1f})"
    else:             return 0,  f"Tehlikeli ({rsi:.1f})"


def puan_macd(macd, sinyal, histogram, onceki_hist):
    if any(np.isnan(v) for v in [macd, sinyal, histogram, onceki_hist]):
        return 5, "Veri yok"
    pozitif  = macd > sinyal
    hist_poz = histogram > 0
    hist_art = histogram > onceki_hist
    if pozitif and hist_poz and hist_art: return 15, "Güçlü Momentum ✓ ↑"
    elif pozitif and hist_poz:            return 10, "Pozitif (zayıflıyor)"
    elif pozitif:                         return 7,  "Üstte ama dikkat"
    elif hist_art:                        return 5,  "Dönüş Sinyali?"
    else:                                 return 0,  "Negatif Momentum ↓"


def puan_hacim(h5, h20):
    if h20 == 0 or np.isnan(h5) or np.isnan(h20): return 3, "Veri yok"
    oran = h5 / h20
    if oran > 2.0:   return 10, f"Çok Yüksek ({oran:.1f}x)"
    elif oran > 1.5: return 8,  f"Yüksek ({oran:.1f}x)"
    elif oran > 1.0: return 6,  f"Ortalama Üstü ({oran:.1f}x)"
    elif oran > 0.7: return 3,  f"Normal ({oran:.1f}x)"
    else:            return 0,  f"Düşük ({oran:.1f}x)"


def puan_atr(atr, fiyat):
    if fiyat <= 0 or np.isnan(atr) or np.isnan(fiyat): return 2, "Veri yok"
    vlt = (atr / fiyat) * 100
    if vlt < 1:      return 0, f"Hareketsiz (%{vlt:.1f})"
    elif vlt < 2:    return 2, f"Düşük (%{vlt:.1f})"
    elif vlt < 5:    return 5, f"İdeal ✓ (%{vlt:.1f})"
    elif vlt < 8:    return 3, f"Yüksek (%{vlt:.1f})"
    else:            return 1, f"Çok Yüksek (%{vlt:.1f})"


# ─────────────────────────────────────────────────────────────────────────────
# ANA ANALİZ FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────

def hisse_analiz_et(ticker: str) -> dict | None:
    try:
        hisse = yf.Ticker(ticker)
        df = hisse.history(period="1y", interval="1d")

        if df is None or len(df) < 50:
            return None

        kapanis = df['Close']
        son_fiyat = kapanis.iloc[-1]
        if son_fiyat <= 0:
            return None

        ma50  = hesapla_ma(kapanis, 50)
        ma200 = hesapla_ma(kapanis, 200)
        rsi   = hesapla_rsi(kapanis)
        macd_val, sinyal_val, hist_val, onceki_hist = hesapla_macd(kapanis)
        atr_val = hesapla_atr(df)

        hacim = df['Volume']
        h5  = hacim.tail(5).mean()
        h20 = hacim.tail(20).mean()

        # Temel veriler
        try:
            info = hisse.info
            pddd   = info.get('priceToBook', np.nan)
            fk     = info.get('trailingPE', np.nan)
            sektor = info.get('sector', 'Bilinmiyor')
            buyume = info.get('earningsQuarterlyGrowth', np.nan)
            if buyume is not None and not (isinstance(buyume, float) and np.isnan(buyume)):
                buyume = float(buyume) * 100
            else:
                yillik = info.get('earningsGrowth', np.nan)
                buyume = float(yillik) * 100 if yillik and not (isinstance(yillik, float) and np.isnan(yillik)) else np.nan
        except Exception:
            pddd, fk, sektor, buyume = np.nan, np.nan, "Bilinmiyor", np.nan

        # Puanlar
        p_pddd, a_pddd   = puan_pddd(pddd)
        p_fk,   a_fk     = puan_fk(fk)
        p_kar,  a_kar    = puan_kar_buyumesi(buyume)
        p_trend, a_trend, trend_gecti = puan_trend(son_fiyat, ma50, ma200)
        p_rsi,  a_rsi    = puan_rsi(rsi)
        p_macd, a_macd   = puan_macd(macd_val, sinyal_val, hist_val, onceki_hist)
        p_hacim,a_hacim  = puan_hacim(h5, h20)
        p_atr,  a_atr    = puan_atr(atr_val, son_fiyat)

        temel   = p_pddd + p_fk + p_kar
        teknik  = (p_trend + p_rsi + p_macd + p_hacim + p_atr) if trend_gecti else 0
        toplam  = temel + teknik

        return {
            "Ticker": ticker.replace(".IS", ""),
            "Fiyat": round(son_fiyat, 2),
            "MA50":  round(ma50, 2)  if not np.isnan(ma50)  else None,
            "MA200": round(ma200, 2) if not np.isnan(ma200) else None,
            "RSI":   round(rsi, 1),
            "PD/DD": round(float(pddd), 2) if pddd and not (isinstance(pddd, float) and np.isnan(pddd)) else None,
            "F/K":   round(float(fk), 1)   if fk   and not (isinstance(fk, float)   and np.isnan(fk))   else None,
            "Sektör": sektor,
            "Trend Geçti": "✅ Evet" if trend_gecti else "❌ Hayır",
            "P_PDDD": p_pddd, "A_PDDD": a_pddd,
            "P_FK":   p_fk,   "A_FK":   a_fk,
            "P_Kar":  p_kar,  "A_Kar":  a_kar,
            "P_Trend":p_trend,"A_Trend":a_trend,
            "P_RSI":  p_rsi,  "A_RSI":  a_rsi,
            "P_MACD": p_macd, "A_MACD": a_macd,
            "P_Hacim":p_hacim,"A_Hacim":a_hacim,
            "P_ATR":  p_atr,  "A_ATR":  a_atr,
            "Temel":  temel,
            "Teknik": teknik,
            "Toplam": toplam,
        }

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLİT ARAYÜZÜ
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Swing Trade Scanner",
    page_icon="📈",
    layout="wide",
)

# ── Stil ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #1e1e2e; border-radius: 12px;
        padding: 1rem; text-align: center;
        border: 1px solid #333;
    }
    .al-badge {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: 700; font-size: 0.85rem;
    }
    .bekle-badge {
        background: #444; color: #aaa; padding: 4px 12px;
        border-radius: 20px; font-size: 0.85rem;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
    }
</style>
""", unsafe_allow_html=True)

# ── Başlık ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📈 BIST Swing Trade Scanner</p>', unsafe_allow_html=True)
st.caption(f"1 Aylık Vade · 100 Puan Sistemi · Temel %40 + Teknik %60 · {datetime.now().strftime('%d.%m.%Y')}")
st.divider()

# ── Kenar Çubuğu (Ayarlar) ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Tarama Ayarları")

    min_puan = st.slider(
        "AL Eşiği (Minimum Puan)", 
        min_value=50, max_value=90, value=70, step=5,
        help="Bu puanın üzerindeki hisseler AL listesine girer."
    )

    st.markdown("---")
    st.subheader("📋 Hisse Listesi")
    liste_secimi = st.radio(
        "Hangi listeyi tara?",
        ["Hazır Liste (Hızlı)", "Özel Liste"],
        help="Özel liste seçersen aşağıya kendi hisselerini girebilirsin."
    )

    if liste_secimi == "Özel Liste":
        ozel_input = st.text_area(
            "Hisse kodlarını virgülle gir (örn: THYAO, GARAN, ASELS)",
            height=150,
            placeholder="THYAO, GARAN, ASELS, EREGL"
        )
        secili_liste = [t.strip().upper() + ".IS" for t in ozel_input.split(",") if t.strip()]
        if not secili_liste:
            st.warning("En az bir hisse kodu gir.")
    else:
        secili_liste = BIST_LISTESI

    st.markdown(f"**Taranacak hisse:** `{len(secili_liste)}`")
    st.markdown("---")

    st.subheader("📊 Puan Dağılımı")
    st.markdown("""
    **Temel Analiz (40 puan)**
    - PD/DD → 15p
    - F/K   → 15p
    - Kar Büyümesi → 10p

    **Teknik Analiz (60 puan)**
    - Trend (MA50/200) → 15p *(Zorunlu)*
    - RSI → 15p
    - MACD → 15p
    - Hacim → 10p
    - ATR → 5p
    """)

# ── Tarama Butonu ─────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    tara_btn = st.button("🚀 Taramayı Başlat", type="primary", use_container_width=True)

with col_info:
    st.info("⏱ Her hisse yaklaşık 0.3 saniye sürer. Hazır liste ~20 saniyede tamamlanır.")

# ── Tarama ────────────────────────────────────────────────────────────────────
if tara_btn:
    if not secili_liste:
        st.error("Lütfen önce hisse listesi seç veya özel liste gir.")
        st.stop()

    st.divider()
    
    # Progress bar
    progress_bar  = st.progress(0, text="Tarama başlıyor...")
    durum_yazisi  = st.empty()
    
    sonuclar = []

    for i, ticker in enumerate(secili_liste):
        yuzde = (i + 1) / len(secili_liste)
        progress_bar.progress(yuzde, text=f"Analiz ediliyor: **{ticker}** ({i+1}/{len(secili_liste)})")
        durum_yazisi.caption(f"🔍 {ticker} işleniyor...")

        sonuc = hisse_analiz_et(ticker)
        if sonuc:
            sonuclar.append(sonuc)

        time.sleep(0.3)

    progress_bar.progress(1.0, text="✅ Tarama tamamlandı!")
    durum_yazisi.empty()

    if not sonuclar:
        st.error("Hiçbir hisseden veri çekilemedi. İnternet bağlantını kontrol et.")
        st.stop()

    df = pd.DataFrame(sonuclar).sort_values("Toplam", ascending=False).reset_index(drop=True)
    al_listesi = df[df["Toplam"] >= min_puan]

    # ── Özet Metrikler ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Tarama Özeti")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Taranan Hisse",   len(df))
    m2.metric("Trend Filtresi Geçen", (df["Trend Geçti"] == "✅ Evet").sum())
    m3.metric(f"AL Listesi ({min_puan}+)", len(al_listesi))
    m4.metric("Ortalama Puan",   f"{df['Toplam'].mean():.1f}")
    m5.metric("En Yüksek Puan",  f"{df['Toplam'].max():.0f}")

    # ── AL Listesi ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"⭐ AL Listesi — {min_puan} Puan ve Üzeri")

    if al_listesi.empty:
        st.warning(f"Şu an {min_puan} puan ve üzeri hisse bulunamadı. Eşiği düşürmeyi dene.")
    else:
        for _, row in al_listesi.iterrows():
            with st.expander(
                f"📈  {row['Ticker']}  |  {row['Fiyat']:.2f} TL  |  🏆 {row['Toplam']:.0f} / 100 puan  |  {row['Sektör']}",
                expanded=False
            ):
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**🔵 Temel Analiz**")
                    st.markdown(f"- PD/DD `{row['P_PDDD']}/15` → {row['A_PDDD']}")
                    st.markdown(f"- F/K `{row['P_FK']}/15` → {row['A_FK']}")
                    st.markdown(f"- Kar Büyümesi `{row['P_Kar']}/10` → {row['A_Kar']}")
                    st.markdown(f"**Temel Toplam: `{row['Temel']}/40`**")

                with c2:
                    st.markdown("**🟢 Teknik Analiz**")
                    st.markdown(f"- Trend `{row['P_Trend']}/15` → {row['A_Trend']}")
                    st.markdown(f"- RSI `{row['P_RSI']}/15` → {row['A_RSI']}")
                    st.markdown(f"- MACD `{row['P_MACD']}/15` → {row['A_MACD']}")
                    st.markdown(f"- Hacim `{row['P_Hacim']}/10` → {row['A_Hacim']}")
                    st.markdown(f"- ATR `{row['P_ATR']}/5` → {row['A_ATR']}")
                    st.markdown(f"**Teknik Toplam: `{row['Teknik']}/60`**")

                # Puan görseli
                puan_data = {
                    "Kategori": ["PD/DD", "F/K", "Kar Büyümesi", "Trend", "RSI", "MACD", "Hacim", "ATR"],
                    "Puan":     [row['P_PDDD'], row['P_FK'], row['P_Kar'],
                                 row['P_Trend'], row['P_RSI'], row['P_MACD'],
                                 row['P_Hacim'], row['P_ATR']],
                    "Maks":     [15, 15, 10, 15, 15, 15, 10, 5],
                }
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=puan_data["Kategori"],
                    y=puan_data["Maks"],
                    name="Maksimum",
                    marker_color="rgba(255,255,255,0.1)",
                ))
                fig.add_trace(go.Bar(
                    x=puan_data["Kategori"],
                    y=puan_data["Puan"],
                    name="Alınan Puan",
                    marker_color=["#00C9FF" if p/m > 0.6 else "#FFD700" if p/m > 0.3 else "#FF6B6B"
                                  for p, m in zip(puan_data["Puan"], puan_data["Maks"])],
                ))
                fig.update_layout(
                    barmode="overlay",
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=0),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Tüm Hisseler Tablosu ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Tüm Hisseler — Sıralı Tablo")

    gosterilecek = df[[
        "Ticker", "Fiyat", "RSI", "PD/DD", "F/K",
        "Trend Geçti", "Temel", "Teknik", "Toplam", "Sektör"
    ]].copy()

    def renk_puan(val):
        if isinstance(val, (int, float)):
            if val >= 70: return "background-color: #1a4a1a; color: #7fff7f"
            elif val >= 50: return "background-color: #3a3a00; color: #ffff88"
            else: return "background-color: #3a0000; color: #ff9999"
        return ""

    st.dataframe(
        gosterilecek.style.applymap(renk_puan, subset=["Toplam"]),
        use_container_width=True,
        height=500,
    )

    # ── Puan Dağılımı Grafiği ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📉 Puan Dağılımı")

    fig2 = px.histogram(
        df, x="Toplam", nbins=20,
        color_discrete_sequence=["#00C9FF"],
        labels={"Toplam": "Toplam Puan", "count": "Hisse Sayısı"},
    )
    fig2.add_vline(x=min_puan, line_dash="dash", line_color="#92FE9D",
                   annotation_text=f"AL Eşiği ({min_puan})", annotation_position="top right")
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── CSV İndir ────────────────────────────────────────────────────────────
    st.divider()
    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Sonuçları CSV İndir",
        data=csv,
        file_name=f"bist_swing_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=False,
    )

else:
    # ── Karşılama Ekranı ──────────────────────────────────────────────────────
    st.markdown("""
    ### 👋 Nasıl Kullanılır?

    1. Sol menüden **AL eşiğini** ayarla (varsayılan: 70)
    2. **Hazır listeyi** kullan ya da kendi hisselerini gir
    3. **"Taramayı Başlat"** butonuna bas
    4. Sonuçları incele, CSV olarak indir

    ---

    ### 📐 Puan Sistemi Nedir?

    | Kategori | Maks Puan | Temel Mantık |
    |---|---|---|
    | PD/DD | 15 | Defter değerine göre ucuzluk |
    | F/K | 15 | Kazanca göre ucuzluk |
    | Kar Büyümesi | 10 | Çeyreksel/yıllık kar artışı |
    | **Trend (MA50/200)** | **15** | **Zorunlu filtre — altındaysa teknik = 0** |
    | RSI | 15 | 50-65 arası ideal swing bölgesi |
    | MACD | 15 | Pozitif ve artan histogram |
    | Hacim | 10 | Son 5G / 20G ortalaması karşılaştırması |
    | ATR | 5 | %2-5 arası ideal volatilite |

    > ⚠️ **Uyarı:** Bu araç yatırım tavsiyesi değildir. Profesyonel danışmanlık alın.
    """)
