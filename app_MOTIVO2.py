
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="Agrícola Progreso | Control de Inventarios",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

EMPRESA = "COMERCIAL AGRÍCOLA PROGRESO S.A.C."

COLUMNAS = [
    "producto", "familia", "precio_unit", "stock_actual",
    "demanda_mensual_jun", "demanda_mensual_jul",
    "demanda_mensual_ago", "lead_time_dias", "sigma_lt"
]

MESES = ["Junio", "Julio", "Agosto"]

DEMANDA_COLS = [
    "demanda_mensual_jun",
    "demanda_mensual_jul",
    "demanda_mensual_ago"
]

# ==========================================================
# 2. ESTILOS VISUALES
# ==========================================================

st.markdown("""
<style>
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

.main-title {
    font-size: clamp(1.6rem, 3vw, 2.5rem);
    font-weight: 800;
    line-height: 1.5;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    padding: 0.5rem 0;
    overflow: visible;
    overflow-wrap: anywhere;
}

.subtitle {
    color: #8492A6;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.18);
    padding: 16px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    font-weight: 750;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
}

h1, h2, h3 {
    line-height: 1.4 !important;
    overflow-wrap: anywhere;
}

button {
    border-radius: 8px !important;
}

@media(max-width: 768px) {
    .block-container {
        padding-top: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. DATOS DEMOSTRATIVOS
# ==========================================================

DATOS_DEMO = pd.DataFrame([
    {
        "producto": "Fertilizante NPK",
        "familia": "Linea 1",
        "precio_unit": 85,
        "stock_actual": 120,
        "demanda_mensual_jun": 180,
        "demanda_mensual_jul": 210,
        "demanda_mensual_ago": 195,
        "lead_time_dias": 7,
        "sigma_lt": 1.5
    },
    {
        "producto": "Semillas certificadas",
        "familia": "Linea 2",
        "precio_unit": 45,
        "stock_actual": 75,
        "demanda_mensual_jun": 100,
        "demanda_mensual_jul": 120,
        "demanda_mensual_ago": 110,
        "lead_time_dias": 5,
        "sigma_lt": 1.0
    },
    {
        "producto": "Insumo agrícola C",
        "familia": "Linea 3",
        "precio_unit": 60,
        "stock_actual": 40,
        "demanda_mensual_jun": 90,
        "demanda_mensual_jul": 105,
        "demanda_mensual_ago": 120,
        "lead_time_dias": 10,
        "sigma_lt": 2.0
    }
])

# ==========================================================
# 4. VALIDACIÓN DE DATOS
# ==========================================================

def validar_datos(datos):

    df = datos.copy()

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    faltantes = [
        c for c in COLUMNAS
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas: " + ", ".join(faltantes)
        )

    df = df[COLUMNAS].copy()

    df = df.dropna(how="all")
    df = df.dropna(subset=["producto"])

    if df.empty:
        raise ValueError("No existen productos registrados.")

    df["producto"] = df["producto"].astype(str).str.strip()
    df["familia"] = df["familia"].fillna("").astype(str).str.strip()

    if (df["producto"] == "").any():
        raise ValueError("Existen productos sin nombre.")

    if (df["familia"] == "").any():
        raise ValueError("Existen productos sin línea asignada.")

    if df["producto"].duplicated().any():
        raise ValueError("No se permiten productos duplicados.")

    if df["familia"].nunique() > 3:
        raise ValueError("El sistema admite máximo tres líneas de productos.")

    numericas = [
        c for c in COLUMNAS
        if c not in ["producto", "familia"]
    ]

    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numericas].isna().any().any():
        raise ValueError("Hay datos numéricos vacíos o inválidos.")

    valores = df[numericas].to_numpy(dtype=float)

    if not np.isfinite(valores).all():
        raise ValueError("Los datos deben ser números finitos.")

    if (df[numericas] < 0).any().any():
        raise ValueError("No se permiten valores negativos.")

    if (df["lead_time_dias"] <= 0).any():
        raise ValueError("El tiempo de entrega debe ser mayor que cero.")

    return df.reset_index(drop=True)

# ==========================================================
# 5. CÁLCULOS DE INVENTARIO
# ==========================================================

def calcular_inventario(datos, nivel_servicio):

    df = validar_datos(datos)

    factores_z = {
        90: 1.282,
        95: 1.645,
        97.5: 1.960,
        99: 2.326
    }

    z = factores_z[nivel_servicio]

    # Demanda promedio mensual
    df["demanda_prom"] = df[DEMANDA_COLS].mean(axis=1)

    # Variabilidad de la demanda mensual
    df["sigma_mensual"] = (
        df[DEMANDA_COLS].std(axis=1, ddof=1).fillna(0)
    )

    # Valor anual estimado de consumo
    df["valor_anual"] = (
        df["demanda_prom"] * 12 * df["precio_unit"]
    )

    # ======================================================
    # CLASIFICACIÓN ABC
    # ======================================================

    df = df.sort_values(
        "valor_anual",
        ascending=False
    ).reset_index(drop=True)

    total_anual = df["valor_anual"].sum()

    if total_anual > 0:

        df["pct_valor"] = df["valor_anual"] / total_anual

        df["pct_acum"] = df["pct_valor"].cumsum()

        anterior = df["pct_acum"].shift(
            1, fill_value=0
        )

        df["clase"] = np.select(
            [
                anterior < 0.80,
                anterior < 0.95
            ],
            ["A", "B"],
            default="C"
        )

    else:

        df["pct_valor"] = 0
        df["pct_acum"] = 0
        df["clase"] = "C"

    # ======================================================
    # DEMANDA DIARIA
    # ======================================================

    df["demanda_diaria"] = df["demanda_prom"] / 30

    # Conversión de desviación mensual a diaria
    df["sigma_diaria"] = (
        df["sigma_mensual"] / np.sqrt(30)
    )

    # ======================================================
    # STOCK DE SEGURIDAD
    # ======================================================

    var_demanda = (
        df["lead_time_dias"] *
        df["sigma_diaria"] ** 2
    )

    var_lead_time = (
        df["demanda_diaria"] ** 2 *
        df["sigma_lt"] ** 2
    )

    df["SS"] = np.ceil(
        z * np.sqrt(var_demanda + var_lead_time)
    ).astype(int)

    # ======================================================
    # PUNTO DE REORDEN
    # ======================================================

    df["demanda_lead_time"] = (
        df["demanda_diaria"] *
        df["lead_time_dias"]
    )

    df["ROP"] = np.ceil(
        df["demanda_lead_time"] + df["SS"]
    ).astype(int)

    # ======================================================
    # ALERTAS Y REPOSICIÓN
    # ======================================================

    df["stock_disponible"] = (
        df["stock_actual"] - df["ROP"]
    )

    df["estado_stock"] = np.select(
        [
            df["stock_actual"] <= df["SS"],
            df["stock_actual"] <= df["ROP"]
        ],
        [
            "Stock crítico",
            "Reponer"
        ],
        default="Stock suficiente"
    )

    df["reponer_unidades"] = np.maximum(
        df["ROP"] - df["stock_actual"], 0
    ).astype(int)

    df["valor_inventario"] = (
        df["stock_actual"] * df["precio_unit"]
    )

    return df

# ==========================================================
# 6. FUNCIONES AUXILIARES
# ==========================================================

def dinero(valor):
    return f"S/ {valor:,.2f}"


def crear_excel(datos):

    salida = BytesIO()

    with pd.ExcelWriter(
        salida,
        engine="openpyxl"
    ) as writer:

        datos.to_excel(
            writer,
            index=False,
            sheet_name="Resultados"
        )

    salida.seek(0)

    return salida.getvalue()


# ==========================================================
# 7. INICIALIZAR SESIÓN
# ==========================================================

if "datos" not in st.session_state:

    st.session_state["datos"] = DATOS_DEMO.copy()

if "archivo" not in st.session_state:

    st.session_state["archivo"] = "Datos demostrativos"

if "servicio" not in st.session_state:

    st.session_state["servicio"] = 95

# ==========================================================
# 8. MENÚ LATERAL
# ==========================================================

with st.sidebar:

    st.markdown("## 🏢 AGRÍCOLA PROGRESO")

    st.caption("Sistema inteligente de inventarios")

    st.divider()

    pagina = st.radio(
        "MENÚ PRINCIPAL",
        [
            "Dashboard Gerencial",
            "Cargar Excel",
            "Base de Datos",
            "Indicadores de Demanda",
            "Análisis ABC",
            "Stock de Seguridad y ROP",
            "Reporte Final"
        ]
    )

    st.divider()

    servicio = st.selectbox(
        "Nivel de servicio al cliente",
        [90, 95, 97.5, 99],
        index=[90, 95, 97.5, 99].index(
            st.session_state["servicio"]
        ),
        format_func=lambda x: f"{x}%"
    )

    st.session_state["servicio"] = servicio

    st.caption(
        "El nivel de servicio modifica dinámicamente el factor Z."
    )

# ==========================================================
# 9. PROCESAMIENTO
# ==========================================================

try:

    df = calcular_inventario(
        st.session_state["datos"],
        st.session_state["servicio"]
    )

except Exception as error:

    st.error(f"Error de procesamiento: {error}")
    st.stop()

# ==========================================================
# 10. ENCABEZADO PRINCIPAL
# ==========================================================

st.markdown(
    """
    <div class="main-title">
        📦 Gestión y Control Inteligente de Inventarios
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">{EMPRESA} | '
    f'Fuente: {st.session_state["archivo"]}</div>',
    unsafe_allow_html=True
)

# ==========================================================
# 11. DASHBOARD GERENCIAL
# ==========================================================

if pagina == "Dashboard Gerencial":

    st.subheader("📊 Panel de control estratégico")

    valor_total = df["valor_inventario"].sum()

    valor_anual = df["valor_anual"].sum()

    unidades = df["stock_actual"].sum()

    demanda = df["demanda_prom"].sum()

    alertas = (df["estado_stock"] != "Stock suficiente").sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Productos analizados",
        len(df)
    )

    c2.metric(
        "Capital invertido",
        dinero(valor_total)
    )

    c3.metric(
        "Unidades disponibles",
        f"{unidades:,.0f}"
    )

    c4.metric(
        "Productos con alerta",
        int(alertas)
    )

    st.divider()

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Demanda mensual",
        f"{demanda:,.1f} unidades"
    )

    c6.metric(
        "Valor anual de consumo",
        dinero(valor_anual)
    )

    c7.metric(
        "Nivel de servicio",
        f"{servicio}%"
    )

    st.divider()

    # GRÁFICO 1
    izquierda, derecha = st.columns(2)

    with izquierda:

        fig1 = px.bar(
            df,
            x="producto",
            y="valor_inventario",
            color="familia",
            text_auto=".2s",
            title="Capital invertido por producto",
            labels={
                "producto": "Producto",
                "valor_inventario": "Capital (S/)",
                "familia": "Línea"
            }
        )

        fig1.update_layout(
            xaxis_title="Producto",
            yaxis_tickformat=",.0f"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # GRÁFICO 2
    with derecha:

        comparacion = df.melt(
            id_vars=["producto"],
            value_vars=["stock_actual", "ROP"],
            var_name="indicador",
            value_name="unidades"
        )

        comparacion["indicador"] = comparacion["indicador"].replace({
            "stock_actual": "Stock actual",
            "ROP": "Punto de reorden"
        })

        fig2 = px.bar(
            comparacion,
            x="producto",
            y="unidades",
            color="indicador",
            barmode="group",
            text_auto=True,
            title="Stock actual vs punto de reorden"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # GRÁFICO 3: DEMANDA LINEAL

    st.subheader("📈 Evolución de la demanda")

    demanda_larga = df.melt(
        id_vars=["producto"],
        value_vars=DEMANDA_COLS,
        var_name="mes",
        value_name="unidades"
    )

    mapa_meses = dict(zip(DEMANDA_COLS, MESES))

    demanda_larga["mes"] = demanda_larga["mes"].map(mapa_meses)

    fig3 = px.line(
        demanda_larga,
        x="mes",
        y="unidades",
        color="producto",
        markers=True,
        category_orders={"mes": MESES},
        title="Tendencia mensual por producto"
    )

    fig3.update_traces(
        line=dict(width=3),
        marker=dict(size=10)
    )

    fig3.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # ALERTAS

    st.divider()

    st.subheader("🚨 Alertas automáticas")

    alertas_df = df[
        df["estado_stock"] != "Stock suficiente"
    ]

    if alertas_df.empty:

        st.success(
            "Todos los productos mantienen stock superior al ROP."
        )

    else:

        st.warning(
            f"Se identificaron {len(alertas_df)} productos que requieren atención."
        )

        st.dataframe(
            alertas_df[
                [
                    "producto",
                    "stock_actual",
                    "SS",
                    "ROP",
                    "reponer_unidades",
                    "estado_stock"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )

# ==========================================================
# 12. CARGAR EXCEL
# ==========================================================

elif pagina == "Cargar Excel":

    st.subheader("📂 Importación de datos")

    st.write(
        "Actualiza la información de inventario mediante un archivo Excel."
    )

    st.info(
        "El archivo debe incluir las nueve columnas de la plantilla."
    )

    st.download_button(
        "⬇ Descargar plantilla Excel",
        data=crear_excel(DATOS_DEMO),
        file_name="plantilla_agricola_progreso.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    archivo = st.file_uploader(
        "Seleccionar archivo Excel",
        type=["xlsx"]
    )

    if archivo is not None:

        try:

            datos_cargados = pd.read_excel(archivo)

            st.subheader("Vista previa")

            st.dataframe(
                datos_cargados.head(10),
                use_container_width=True
            )

            if st.button(
                "VALIDAR E IMPORTAR DATOS",
                type="primary"
            ):

                nuevos_datos = validar_datos(datos_cargados)

                st.session_state["datos"] = nuevos_datos

                st.session_state["archivo"] = archivo.name

                st.success(
                    "¡Base de datos actualizada correctamente!"
                )

                st.rerun()

        except Exception as error:

            st.error(f"No se pudo importar el archivo: {error}")

    st.divider()

    if st.button("Restaurar datos demostrativos"):

        st.session_state["datos"] = DATOS_DEMO.copy()

        st.session_state["archivo"] = "Datos demostrativos"

        st.rerun()

# ==========================================================
# 13. BASE DE DATOS
# ==========================================================

elif pagina == "Base de Datos":

    st.subheader("📋 Registro maestro de inventarios")

    st.write(
        "Información original y resultados calculados automáticamente."
    )

    st.dataframe(
        df[
            COLUMNAS + [
                "demanda_prom",
                "sigma_mensual",
                "valor_anual",
                "clase",
                "SS",
                "ROP",
                "estado_stock"
            ]
        ].style.format({
            "precio_unit": "S/ {:.2f}",
            "valor_anual": "S/ {:,.2f}",
            "demanda_prom": "{:.2f}",
            "sigma_mensual": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Los datos iniciales son demostrativos. "
        "Reemplázalos por los registros reales de la empresa."
    )

# ==========================================================
# 14. INDICADORES DE DEMANDA
# ==========================================================

elif pagina == "Indicadores de Demanda":

    st.subheader("📈 Análisis de comportamiento de la demanda")

    seleccion = st.multiselect(
        "Seleccionar productos",
        df["producto"].tolist(),
        default=df["producto"].tolist()
    )

    if not seleccion:

        st.info("Selecciona al menos un producto.")

    else:

        filtrado = df[
            df["producto"].isin(seleccion)
        ].copy()

        datos = filtrado.melt(
            id_vars=["producto"],
            value_vars=DEMANDA_COLS,
            var_name="mes",
            value_name="demanda"
        )

        datos["mes"] = datos["mes"].map(
            dict(zip(DEMANDA_COLS, MESES))
        )

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Demanda acumulada",
            f"{filtrado[DEMANDA_COLS].to_numpy().sum():,.0f}"
        )

        k2.metric(
            "Promedio mensual",
            f"{filtrado['demanda_prom'].mean():,.1f}"
        )

        k3.metric(
            "Mayor demanda promedio",
            f"{filtrado['demanda_prom'].max():,.1f}"
        )

        st.divider()

        # GRÁFICO LINEAL

        fig1 = px.line(
            datos,
            x="mes",
            y="demanda",
            color="producto",
            markers=True,
            category_orders={"mes": MESES},
            title="Evolución de demanda mensual"
        )

        fig1.update_traces(
            line=dict(width=3),
            marker=dict(size=10)
        )

        fig1.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # GRÁFICO DE BARRAS

        fig2 = px.bar(
            datos,
            x="mes",
            y="demanda",
            color="producto",
            barmode="group",
            text_auto=True,
            category_orders={"mes": MESES},
            title="Comparación de demanda por mes"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # GRÁFICO DE VARIABILIDAD

        fig3 = px.bar(
            filtrado,
            x="producto",
            y="sigma_mensual",
            color="familia",
            text_auto=".2f",
            title="Variabilidad de la demanda mensual",
            labels={
                "sigma_mensual": "Desviación estándar"
            }
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.subheader("Resumen estadístico")

        st.dataframe(
            filtrado[
                [
                    "producto",
                    "demanda_prom",
                    "sigma_mensual",
                    "demanda_diaria"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "El análisis utiliza tres meses históricos. "
            "Las líneas representan demanda observada, no pronósticos futuros."
        )

# ==========================================================
# 15. ANÁLISIS ABC
# ==========================================================

elif pagina == "Análisis ABC":

    st.subheader("📊 Clasificación ABC automática")

    st.write(
        "Clasificación según el valor anual estimado de consumo."
    )

    a, b, c = st.columns(3)

    a.metric(
        "Productos A",
        int((df["clase"] == "A").sum())
    )

    b.metric(
        "Productos B",
        int((df["clase"] == "B").sum())
    )

    c.metric(
        "Productos C",
        int((df["clase"] == "C").sum())
    )

    st.divider()

    # PARETO

    pareto = df.sort_values(
        "valor_anual",
        ascending=False
    ).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=pareto["producto"],
            y=pareto["valor_anual"],
            name="Valor anual",
            text=[
                f"S/ {v:,.0f}"
                for v in pareto["valor_anual"]
            ],
            textposition="auto"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pareto["producto"],
            y=pareto["pct_acum"] * 100,
            name="% acumulado",
            mode="lines+markers+text",
            text=[
                f"{v:.1f}%"
                for v in pareto["pct_acum"] * 100
            ],
            textposition="top center",
            yaxis="y2",
            line=dict(width=3)
        )
    )

    fig.update_layout(
        title="Diagrama de Pareto - Análisis ABC",
        xaxis_title="Productos",
        yaxis=dict(
            title="Valor anual (S/)",
            tickformat=",.0f"
        ),
        yaxis2=dict(
            title="Porcentaje acumulado",
            overlaying="y",
            side="right",
            range=[0, 110],
            ticksuffix="%"
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detalle de clasificación")

    st.dataframe(
        df[
            [
                "producto",
                "familia",
                "valor_anual",
                "pct_valor",
                "pct_acum",
                "clase"
            ]
        ].style.format({
            "valor_anual": "S/ {:,.2f}",
            "pct_valor": "{:.1%}",
            "pct_acum": "{:.1%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Clase A: mayor prioridad de control. "
        "Clase B: prioridad intermedia. "
        "Clase C: control simplificado."
    )

# ==========================================================
# 16. STOCK DE SEGURIDAD Y ROP
# ==========================================================

elif pagina == "Stock de Seguridad y ROP":

    st.subheader("📦 Gestión dinámica del inventario")

    z = {
        90: 1.282,
        95: 1.645,
        97.5: 1.960,
        99: 2.326
    }[servicio]

    st.metric(
        "Nivel de servicio seleccionado",
        f"{servicio}%"
    )

    st.metric(
        "Factor de seguridad Z",
        f"{z:.3f}"
    )

    st.markdown("### Fórmulas utilizadas")

    st.latex(
        r"SS=Z\sqrt{L\sigma_d^2+\bar d^2\sigma_{LT}^2}"
    )

    st.latex(
        r"ROP=(\bar d \times L)+SS"
    )

    st.caption(
        "SS: stock de seguridad. "
        "L: tiempo medio de entrega. "
        "σd: desviación estándar de la demanda diaria. "
        "σLT: desviación estándar del tiempo de entrega. "
        "d̄: demanda diaria promedio."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.bar(
            df,
            x="producto",
            y="SS",
            color="familia",
            text_auto=True,
            title="Stock de seguridad por producto"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                x=df["producto"],
                y=df["stock_actual"],
                name="Stock actual"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=df["producto"],
                y=df["ROP"],
                name="Punto de reorden",
                mode="lines+markers+text",
                text=df["ROP"],
                textposition="top center",
                line=dict(width=3)
            )
        )

        fig2.update_layout(
            title="Stock disponible vs ROP",
            xaxis_title="Producto",
            yaxis_title="Unidades",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.subheader("Tabla de reposición")

    st.dataframe(
        df[
            [
                "producto",
                "stock_actual",
                "demanda_diaria",
                "lead_time_dias",
                "SS",
                "ROP",
                "estado_stock",
                "reponer_unidades"
            ]
        ].style.format({
            "demanda_diaria": "{:.2f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "La cantidad sugerida representa la diferencia entre el ROP "
        "y el stock físico. Para emitir órdenes reales se deben considerar "
        "también pedidos en tránsito y órdenes pendientes."
    )

# ==========================================================
# 17. REPORTE FINAL
# ==========================================================

elif pagina == "Reporte Final":

    st.subheader("📑 Informe estratégico de inventarios")

    alertas = df[
        df["estado_stock"] != "Stock suficiente"
    ]

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Productos analizados",
        len(df)
    )

    r2.metric(
        "Productos con alerta",
        len(alertas)
    )

    r3.metric(
        "Capital invertido",
        dinero(df["valor_inventario"].sum())
    )

    st.divider()

    st.markdown(f"**Empresa:** {EMPRESA}")

    st.markdown(f"**Nivel de servicio:** {servicio}%")

    st.markdown(
        f"**Archivo analizado:** {st.session_state['archivo']}"
    )

    st.subheader("Recomendaciones automáticas")

    for _, fila in df.iterrows():

        if fila["estado_stock"] == "Stock crítico":

            st.error(
                f"🚨 {fila['producto']}: stock crítico. "
                f"Revisar inmediatamente. "
                f"Reposición sugerida: {fila['reponer_unidades']} unidades."
            )

        elif fila["estado_stock"] == "Reponer":

            st.warning(
                f"⚠️ {fila['producto']}: alcanzó el punto de reorden. "
                f"Reposición sugerida: {fila['reponer_unidades']} unidades."
            )

        else:

            st.success(
                f"✅ {fila['producto']}: mantiene stock suficiente."
            )

    st.divider()

    st.subheader("Exportar resultados")

    excel = crear_excel(df)

    st.download_button(
        "⬇ Descargar reporte completo en Excel",
        data=excel,
        file_name="reporte_agricola_progreso.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

    csv = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇ Descargar reporte en CSV",
        data=csv,
        file_name="reporte_agricola_progreso.csv",
        mime="text/csv"
    )

# ==========================================================
# 18. PIE DE PÁGINA
# ==========================================================

st.divider()

st.caption(
    f"© {EMPRESA} | Sistema de Gestión y Control de Inventarios "
    "| Análisis ABC | Stock de Seguridad | Punto de Reorden"
)
