import pandas as pd
import numpy as np
import unicodedata
import os
import matplotlib.pyplot as plt


class CONTENEDOR:

    def cargar_excel(self, path):
        df_incidentes = pd.read_csv(path)
        return df_incidentes
    

    def analizar(self, df):

        print("INFORMACIÓN GENERAL")
        print("=" * 60)

        print(f"Filas: {df.shape[0]}")
        print(f"Columnas: {df.shape[1]}")

        print("\nNombres de columnas:")
        print(df.columns.tolist())

        print("\nTipos de datos:")
        print(df.dtypes.to_string())

        print("\nPrimeras 5 filas:")
        print(df.head().to_string())

        print("\nÚltimas 5 filas:")
        print(df.tail().to_string())

        print("\n--- INFO DEL DATAFRAME ---")
        df.info()

        return df


    def analizar_nulos(self, df):

        if df is None:
            raise ValueError("El DataFrame recibido es None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Se esperaba un DataFrame y se recibió: {type(df)}")

        nulos = df.isnull()

        total_filas = len(df)
        total_columnas = len(df.columns)
        total_nulos = nulos.sum().sum()

        filas_con_nulos = nulos.any(axis=1).sum()

        porcentaje_filas_afectadas = round(
            (filas_con_nulos / total_filas) * 100, 2
        )

        resumen = pd.DataFrame({
            "Nulos": nulos.sum(),
            "% Nulos": round(nulos.mean() * 100, 2)
        })

        resumen = resumen[resumen["Nulos"] > 0]

        if len(resumen) > 0:
            resumen = resumen.sort_values(
                by="% Nulos",
                ascending=False
            )

        print("\n" + "=" * 70)
        print("ANÁLISIS DE VALORES NULOS")
        print("=" * 70)

        print(f"Total registros: {total_filas:,}")
        print(f"Total columnas: {total_columnas:,}")
        print(f"Total valores nulos: {total_nulos:,}")

        if total_nulos == 0:
            print("\n✅ No se encontraron valores nulos.")
            print("=" * 70)
            return df

        print("\nColumnas con valores nulos:")
        print(resumen.to_string())

        columnas_vacias = resumen[
            resumen["% Nulos"] == 100
        ]

        print("\nColumnas completamente vacías:")

        if len(columnas_vacias) > 0:
            print(columnas_vacias.index.tolist())
        else:
            print("Ninguna")

        columnas_mas_50 = resumen[
            resumen["% Nulos"] > 50
        ]

        print("\nColumnas con más del 50% de nulos:")

        if len(columnas_mas_50) > 0:
            print(columnas_mas_50.to_string())
        else:
            print("Ninguna")

        print("\nTop 10 columnas con más nulos:")
        print(resumen.head(10).to_string())

        print(f"\nFilas con al menos un nulo: {filas_con_nulos:,}")

        print(
            f"Porcentaje de filas afectadas: "
            f"{porcentaje_filas_afectadas}%"
        )

        print("=" * 70)

        return df


    def normalizar(self, df):

        if df is None:
            raise ValueError("El DataFrame recibido es None")

        def quitar_tildes(texto):
            if pd.isna(texto):
                return texto

            texto = str(texto)

            return "".join(
                c for c in unicodedata.normalize("NFD", texto)
                if unicodedata.category(c) != "Mn"
            )

        nuevas_columnas = []

        for col in df.columns:

            col = quitar_tildes(col)

            col = col.lower().strip()

            col = col.replace(" ", "_")
            col = col.replace("-", "_")
            col = col.replace("/", "_")
            col = col.replace("\\", "_")
            col = col.replace(".", "_")
            col = col.replace(",", "_")
            col = col.replace("(", "")
            col = col.replace(")", "")

            while "__" in col:
                col = col.replace("__", "_")

            nuevas_columnas.append(col)

        df.columns = nuevas_columnas

        columnas_texto = df.select_dtypes(
            include=["object"]
        ).columns

        for col in columnas_texto:

            df[col] = (
                df[col]
                .astype(str)
                .apply(quitar_tildes)
                .str.lower()
                .str.strip()
                .str.replace(r"\s+", "_", regex=True)
            )

        # ==========================================
        # NORMALIZAR VARIABLES NUMÉRICAS
        # ==========================================

        columnas_numericas = df.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns

        for col in columnas_numericas:

            minimo = df[col].min()
            maximo = df[col].max()

            if maximo != minimo:
                df[col] = (df[col] - minimo) / (maximo - minimo)
            else:
                df[col] = 0

        print("=" * 60)
        print("NORMALIZACION COMPLETADA")
        print("=" * 60)

        return df

    def analisis_descriptivo(self, df):

        if df is None:
            raise ValueError("El DataFrame recibido es None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Se esperaba un DataFrame y se recibió: {type(df)}")

        print("\n" + "=" * 80)
        print("ANÁLISIS DESCRIPTIVO GENERAL")
        print("=" * 80)

        print(f"\nTotal de registros: {df.shape[0]:,}")
        print(f"Total de columnas: {df.shape[1]:,}")

        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

        columnas_categoricas = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        print("\nVariables numéricas encontradas:")
        if len(columnas_numericas) > 0:
            print(columnas_numericas)
        else:
            print("No hay variables numéricas.")

        print("\nVariables categóricas encontradas:")
        if len(columnas_categoricas) > 0:
            print(columnas_categoricas)
        else:
            print("No hay variables categóricas.")

        if len(columnas_numericas) > 0:
            print("\n" + "-" * 80)
            print("ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS")
            print("-" * 80)

            descriptivo_numerico = df[columnas_numericas].describe().T.copy()

            descriptivo_numerico["mediana"] = df[columnas_numericas].median()
            descriptivo_numerico["varianza"] = df[columnas_numericas].var()
            descriptivo_numerico["asimetria"] = df[columnas_numericas].skew()
            descriptivo_numerico["curtosis"] = df[columnas_numericas].kurtosis()
            descriptivo_numerico["nulos"] = df[columnas_numericas].isnull().sum()
            descriptivo_numerico["%_nulos"] = round(
                df[columnas_numericas].isnull().mean() * 100, 2
            )

            print(descriptivo_numerico.to_string())

        if len(columnas_categoricas) > 0:
            print("\n" + "-" * 80)
            print("ESTADÍSTICAS DESCRIPTIVAS - VARIABLES CATEGÓRICAS")
            print("-" * 80)

            descriptivo_categorico = pd.DataFrame({
                "variable": columnas_categoricas,
                "valores_unicos": [
                    df[col].nunique(dropna=True)
                    for col in columnas_categoricas
                ],
                "nulos": [
                    df[col].isnull().sum()
                    for col in columnas_categoricas
                ],
                "%_nulos": [
                    round(df[col].isnull().mean() * 100, 2)
                    for col in columnas_categoricas
                ],
                "moda": [
                    df[col].mode(dropna=True).iloc[0]
                    if not df[col].mode(dropna=True).empty
                    else None
                    for col in columnas_categoricas
                ],
                "frecuencia_moda": [
                    df[col].value_counts(dropna=True).iloc[0]
                    if not df[col].value_counts(dropna=True).empty
                    else 0
                    for col in columnas_categoricas
                ]
            })

            print(descriptivo_categorico.to_string(index=False))

            print("\nTop 10 valores por variable categórica:")

            for col in columnas_categoricas:
                print("\n" + "-" * 60)
                print(f"Variable: {col}")
                print("-" * 60)
                print(df[col].value_counts(dropna=False).head(10).to_string())

        if len(columnas_numericas) >= 2:
            print("\n" + "-" * 80)
            print("MATRIZ DE CORRELACIÓN - VARIABLES NUMÉRICAS")
            print("-" * 80)

            matriz_correlacion = df[columnas_numericas].corr()

            print(matriz_correlacion.to_string())

        else:
            print("\nNo hay suficientes variables numéricas para calcular matriz de correlación.")

        print("\n" + "=" * 80)
        print("ANÁLISIS DESCRIPTIVO FINALIZADO")
        print("=" * 80)

        return df
            
        

    def estadistica(self, df):

        if df is None:
            raise ValueError("El DataFrame recibido es None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Se esperaba un DataFrame y se recibió: {type(df)}")

        print("\n" + "=" * 70)
        print("ESTADÍSTICA DESCRIPTIVA GENERAL")
        print("=" * 70)

        print(f"\nTotal de registros: {df.shape[0]:,}")
        print(f"Total de columnas: {df.shape[1]:,}")

        print("\nTipos de datos:")
        print(df.dtypes.to_string())

        columnas_numericas = df.select_dtypes(
            include=[np.number]
        ).columns

        columnas_categoricas = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        print("\nVariables numéricas:")
        print(list(columnas_numericas) if len(columnas_numericas) > 0 else "No hay variables numéricas.")

        print("\nVariables categóricas:")
        print(list(columnas_categoricas) if len(columnas_categoricas) > 0 else "No hay variables categóricas.")

        if len(columnas_numericas) > 0:
            print("\nResumen estadístico de variables numéricas:")
            print(df[columnas_numericas].describe().T.to_string())

        if len(columnas_categoricas) > 0:
            print("\nResumen estadístico de variables categóricas:")
            print(df[columnas_categoricas].describe().T.to_string())

        print("\n" + "=" * 70)
        print("FIN DE LA ESTADÍSTICA DESCRIPTIVA GENERAL")
        print("=" * 70)

        return df
    

    def responder_preguntas_negocio(self, df):
        """
        Genera análisis y visualizaciones para responder las preguntas de negocio.
        Retorna el mismo DataFrame para continuar el flujo.
        """

        if df is None:
            raise ValueError("El DataFrame recibido es None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Se esperaba un DataFrame y se recibió: {type(df)}")

        print("\n" + "=" * 80)
        print("ANÁLISIS DE PREGUNTAS DE NEGOCIO")
        print("=" * 80)

        # ============================================================
        # PREGUNTA 1
        # ¿Qué tipos de ataques generan mayor impacto operativo
        # y financiero?
        # ============================================================

        print("\nPREGUNTA 1")
        print("¿Qué tipos de ataques generan el mayor impacto operativo y financiero?")
        print("-" * 80)

        if "attack_type" in df.columns and "financial_loss" in df.columns:

            impacto_financiero = (
                df.groupby("attack_type")["financial_loss"]
                .sum()
                .sort_values(ascending=False)
            )

            print("\nImpacto financiero por tipo de ataque:")
            print(impacto_financiero.to_string())

            impacto_financiero.head(10).plot(
                kind="bar",
                figsize=(10, 5),
                title="Top 10 tipos de ataque por impacto financiero"
            )

            plt.xlabel("Tipo de ataque")
            plt.ylabel("Pérdida financiera total")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        if "attack_type" in df.columns and "affected_assets" in df.columns:

            impacto_operativo = (
                df.groupby("attack_type")["affected_assets"]
                .sum()
                .sort_values(ascending=False)
            )

            print("\nImpacto operativo por tipo de ataque:")
            print(impacto_operativo.to_string())

            impacto_operativo.head(10).plot(
                kind="bar",
                figsize=(10, 5),
                title="Top 10 tipos de ataque por activos afectados"
            )

            plt.xlabel("Tipo de ataque")
            plt.ylabel("Total de activos afectados")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        # ============================================================
        # PREGUNTA 2
        # ¿Qué tan eficaz es el proceso de respuesta ante incidentes?
        # ============================================================

        print("\nPREGUNTA 2")
        print("¿Qué tan eficaz es el proceso de respuesta ante incidentes?")
        print("-" * 80)

        if "response_time" in df.columns:

            print("\nTiempo promedio de respuesta:")
            print(df["response_time"].mean())

            print("\nTiempo máximo de respuesta:")
            print(df["response_time"].max())

        if "incident_resolution_time" in df.columns:

            print("\nTiempo promedio de resolución:")
            print(df["incident_resolution_time"].mean())

            print("\nTiempo máximo de resolución:")
            print(df["incident_resolution_time"].max())

        if "attack_type" in df.columns and "response_time" in df.columns:

            respuesta_por_ataque = (
                df.groupby("attack_type")["response_time"]
                .mean()
                .sort_values(ascending=False)
            )

            print("\nTiempo promedio de respuesta por tipo de ataque:")
            print(respuesta_por_ataque.to_string())

            respuesta_por_ataque.head(10).plot(
                kind="bar",
                figsize=(10, 5),
                title="Tipos de ataque con mayor tiempo promedio de respuesta"
            )

            plt.xlabel("Tipo de ataque")
            plt.ylabel("Tiempo promedio de respuesta")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        if "attack_type" in df.columns and "incident_resolution_time" in df.columns:

            resolucion_por_ataque = (
                df.groupby("attack_type")["incident_resolution_time"]
                .mean()
                .sort_values(ascending=False)
            )

            print("\nTiempo promedio de resolución por tipo de ataque:")
            print(resolucion_por_ataque.to_string())

            resolucion_por_ataque.head(10).plot(
                kind="bar",
                figsize=(10, 5),
                title="Tipos de ataque con mayor tiempo promedio de resolución"
            )

            plt.xlabel("Tipo de ataque")
            plt.ylabel("Tiempo promedio de resolución")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        # ============================================================
        # PREGUNTA 3
        # ¿Cuáles son los riesgos más relevantes para el negocio?
        # ============================================================

        print("\nPREGUNTA 3")
        print("¿Cuáles son los riesgos más relevantes para el negocio?")
        print("-" * 80)

        if "severity_level" in df.columns:

            severidad = df["severity_level"].value_counts()

            print("\nDistribución por nivel de severidad:")
            print(severidad.to_string())

            severidad.plot(
                kind="bar",
                figsize=(8, 5),
                title="Distribución de incidentes por nivel de severidad"
            )

            plt.xlabel("Nivel de severidad")
            plt.ylabel("Cantidad de incidentes")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        if "attack_type" in df.columns and "severity_level" in df.columns:

            matriz_riesgo = pd.crosstab(
                df["attack_type"],
                df["severity_level"]
            )

            print("\nMatriz de riesgo por tipo de ataque y severidad:")
            print(matriz_riesgo.to_string())

            matriz_riesgo.plot(
                kind="bar",
                stacked=True,
                figsize=(12, 6),
                title="Matriz de riesgo: tipo de ataque vs severidad"
            )

            plt.xlabel("Tipo de ataque")
            plt.ylabel("Cantidad de incidentes")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        if "vulnerability_type" in df.columns:

            vulnerabilidades = df["vulnerability_type"].value_counts()

            print("\nVulnerabilidades más frecuentes:")
            print(vulnerabilidades.head(10).to_string())

            vulnerabilidades.head(10).plot(
                kind="bar",
                figsize=(10, 5),
                title="Top 10 vulnerabilidades más frecuentes"
            )

            plt.xlabel("Tipo de vulnerabilidad")
            plt.ylabel("Cantidad de incidentes")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        print("\n" + "=" * 80)
        print("ANÁLISIS DE NEGOCIO FINALIZADO")
        print("=" * 80)

        return df

    def guardar_excel(self, df, nombre_archivo):
       
        if df is None:
            raise ValueError(
                "El DataFrame recibido es None."
            )

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Se esperaba un DataFrame y se recibió: {type(df)}"
            )

        nombre_base, extension = os.path.splitext(nombre_archivo)

        nuevo_nombre = f"{nombre_base}_1{extension}"

        df.to_excel(
            nuevo_nombre,
            index=False
        )

        print("\n" + "=" * 60)
        print("ARCHIVO EXCEL GUARDADO CORRECTAMENTE")
        print("=" * 60)
        print(f"Archivo: {nuevo_nombre}")
        print(f"Registros: {df.shape[0]:,}")
        print(f"Columnas: {df.shape[1]:,}")
        print("=" * 60)

        return df