import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
CONVERSION = 235.214583
COLORS = {"usa": "#ff4048", "japan": "#7fc4f8", "europe": "#0874d1"}

st.set_page_config(page_title="Exploration des données automobiles", page_icon="🚗", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
[data-testid="stMetricValue"] {font-size: 2rem;}
h1,h2,h3 {color: #303442;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL)
    df["consumption_l_100km"] = CONVERSION / df["mpg"]
    df["weight_kg"] = df["weight"] * 0.45359237
    return df

try:
    df = load_data()
except Exception as exc:
    st.error("Impossible de charger le dataset Auto MPG. Réessayez plus tard.")
    st.exception(exc)
    st.stop()

st.sidebar.header("Filtres")
origin = st.sidebar.selectbox("Origine", ["Toutes"] + sorted(df["origin"].dropna().unique().tolist()))
cylinder_options = sorted(df["cylinders"].dropna().astype(int).unique().tolist())
cylinders = st.sidebar.multiselect("Nombre de cylindres", cylinder_options, default=cylinder_options)
year_min, year_max = int(df["model_year"].min()) + 1900, int(df["model_year"].max()) + 1900
years = st.sidebar.slider("Année du modèle", year_min, year_max, (year_min, year_max))
min_mpg = st.sidebar.slider("Efficacité minimale (mpg)", float(df["mpg"].min()), float(df["mpg"].max()), float(df["mpg"].min()), 0.5)

filtered = df.copy()
if origin != "Toutes":
    filtered = filtered[filtered["origin"] == origin]
filtered = filtered[filtered["cylinders"].isin(cylinders)]
filtered = filtered[filtered["model_year"].between(years[0] - 1900, years[1] - 1900)]
filtered = filtered[filtered["mpg"] >= min_mpg]

display_cols = ["name", "origin", "year", "cylinders", "mpg", "consumption_l_100km", "horsepower", "weight_kg", "acceleration"]
def display_table(frame):
    result = frame.copy()
    result["year"] = result["model_year"] + 1900
    return result[display_cols].round(2)

st.title("🚗 Exploration des données automobiles")
st.caption("Dataset historique Auto MPG · CSV public du dépôt Seaborn")
tab_analysis, tab_ml = st.tabs(["📊 Analyse descriptive", "🤖 Prédiction Random Forest"])

with tab_analysis:
    st.info("mpg élevé = meilleure efficacité. L/100 km faible = consommation moindre.")
    st.header("1 — Indicateurs et données filtrées")
    st.write(f"{len(filtered)} observations sur {len(df)}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Efficacité moyenne", f"{filtered['mpg'].mean():.1f} mpg" if len(filtered) else "—")
    c2.metric("Consommation moyenne", f"{filtered['consumption_l_100km'].mean():.1f} L/100 km" if len(filtered) else "—")
    c3.metric("Puissance moyenne", f"{filtered['horsepower'].mean():.1f} hp" if len(filtered) else "—")
    st.caption("Les moyennes ignorent les valeurs manquantes. La consommation moyenne est la moyenne des conversions individuelles.")
    st.dataframe(display_table(filtered), use_container_width=True, hide_index=True, height=330)

    st.header("2 — Choisir un véhicule")
    if len(filtered):
        vehicle_ids = filtered.index.tolist()
        chosen = st.selectbox("Véhicule", vehicle_ids, format_func=lambda i: f"{df.loc[i, 'name']} — {int(df.loc[i, 'model_year'])+1900} (ligne {i+1})")
        st.dataframe(display_table(df.loc[[chosen]]), use_container_width=True, hide_index=True)
    else:
        st.warning("Aucun véhicule ne correspond aux filtres. Élargissez la sélection.")

    st.header("3 — Comparaisons graphiques")
    if len(filtered):
        counts = filtered.groupby("origin").size().reset_index(name="Nombre")
        fig1 = px.bar(counts, x="origin", y="Nombre", color="origin", title="Nombre d’observations par origine", color_discrete_map=COLORS, labels={"origin": "Origine"})
        fig1.update_layout(legend_title_text="Origine")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(filtered, x="weight_kg", y="mpg", color="origin", hover_name="name",
                          title="Poids et efficacité énergétique", color_discrete_map=COLORS,
                          labels={"weight_kg": "Poids (kg)", "mpg": "Efficacité (mpg)", "origin": "Origine"})
        st.plotly_chart(fig2, use_container_width=True)

        evolution = filtered.groupby("model_year", as_index=False)["mpg"].mean()
        evolution["year"] = evolution["model_year"] + 1900
        fig3 = px.line(evolution, x="year", y="mpg", markers=True, title="Efficacité moyenne par année de modèle",
                       labels={"year": "Année du modèle", "mpg": "Efficacité moyenne (mpg)"})
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.box(filtered, x="origin", y="consumption_l_100km", color="origin",
                      title="Comparaison de la consommation par origine", color_discrete_map=COLORS,
                      labels={"origin": "Origine", "consumption_l_100km": "Consommation (L/100 km)"})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Aucune donnée à représenter.")

FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
NUMERIC = FEATURES[:-1]

@st.cache_resource
def train_model():
    data = df.dropna(subset=["consumption_l_100km"]).copy()
    X = data[FEATURES]
    y = data["consumption_l_100km"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("origin", OneHotEncoder(handle_unknown="ignore"), ["origin"])
    ])
    pipeline = Pipeline([
        ("preprocessing", prep),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
        "R²": r2_score(y_test, predicted)
    }
    return pipeline, metrics, len(X_train), len(X_test)

with tab_ml:
    st.header("Prédiction de la consommation automobile")
    st.write("Modèle Random Forest Regressor entraîné sur les véhicules historiques Auto MPG. Les estimations ne constituent pas des mesures réelles.")
    model, metrics, n_train, n_test = train_model()
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.2f} L/100 km")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f} L/100 km")
    m3.metric("R²", f"{metrics['R²']:.3f}")
    st.caption(f"Évaluation sur {n_test} véhicules de test indépendants ; entraînement sur {n_train} véhicules (séparation 80/20, random_state=42).")

    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            input_origin = st.selectbox("Origine du véhicule", sorted(df["origin"].unique()))
            input_cyl = st.selectbox("Nombre de cylindres", cylinder_options, index=cylinder_options.index(4))
            input_disp = st.number_input("Cylindrée (cubic inches)", min_value=50.0, max_value=500.0, value=150.0, step=5.0)
            input_hp = st.number_input("Puissance (hp)", min_value=30.0, max_value=300.0, value=100.0, step=5.0)
        with b:
            input_weight = st.number_input("Poids (kg)", min_value=500.0, max_value=2500.0, value=1200.0, step=25.0)
            input_accel = st.number_input("Accélération 0–60 mph (secondes)", min_value=8.0, max_value=25.0, value=15.0, step=0.5)
            input_year = st.slider("Année du modèle", year_min, year_max, 1978)
        submit = st.form_submit_button("🔍 Estimer la consommation", type="primary", use_container_width=True)

    if submit:
        inputs = pd.DataFrame([{
            "cylinders": input_cyl, "displacement": input_disp, "horsepower": input_hp,
            "weight": input_weight / 0.45359237, "acceleration": input_accel,
            "model_year": input_year - 1900, "origin": input_origin
        }])
        prediction = float(model.predict(inputs)[0])
        st.success(f"Consommation estimée : {prediction:.2f} L/100 km")
        st.metric("Efficacité équivalente", f"{CONVERSION/prediction:.1f} mpg")
        st.caption("Estimation valable surtout pour des véhicules comparables à ceux du dataset (années 1970–1982).")

    with st.expander("Importance des caractéristiques du modèle"):
        prep = model.named_steps["preprocessing"]
        names = prep.get_feature_names_out()
        importance = pd.DataFrame({
            "Caractéristique": names,
            "Importance": model.named_steps["model"].feature_importances_
        }).sort_values("Importance", ascending=False)
        fig_imp = px.bar(importance, x="Importance", y="Caractéristique", orientation="h", title="Importance des variables")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)
