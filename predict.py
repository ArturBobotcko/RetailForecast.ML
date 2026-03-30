import pandas as pd
import argparse
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Прогноз выбранного показателя с помощью Lasso Regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
            Примеры использования:
            python forecast_lasso.py -f data/finance.xlsx -t "Чистая прибыль, млрд.руб." -p 2030
            python forecast_lasso.py --file finance.xlsx --target "Выручка, млрд.руб." --period 2028
        """
    )
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Путь к файлу с данными (например, "data/finance.xlsx")')
    parser.add_argument('-t', '--target', type=str, required=True,
                        help='Название столбца с прогнозируемым показателем (например, "Чистая прибыль, млрд.руб.")')
    parser.add_argument('-p', '--period', type=int, required=True,
                        help='Год, до которого нужно сделать прогноз (например, 2030)')

    args = parser.parse_args()

    file_path = Path(args.file)
    data = load_data(file_path)
    if data is None:
        return

    year_col = 'Год'
    target_col = args.target.strip()

    if year_col not in data.columns or target_col not in data.columns:
        print(f"Ошибка: Отсутствует колонка '{year_col}' или '{target_col}'")
        return

    if not pd.api.types.is_numeric_dtype(data[target_col]):
        print(f"Ошибка: '{target_col}' должен быть числовым")
        return

    features = [col for col in data.columns if col not in [year_col, target_col] and pd.api.types.is_numeric_dtype(data[col])]

    if not features:
        print("Ошибка: Нет числовых признаков")
        return

    max_historical_year = int(data[year_col].max())
    if args.period <= max_historical_year:
        print(f"Ошибка: Год прогноза должен быть > {max_historical_year}")
        return

    future_years = np.arange(max_historical_year + 1, args.period + 1)

    feature_models = {}
    for feat in features:
        lr = LinearRegression()
        lr.fit(data[year_col].values.reshape(-1, 1), data[feat].values)
        feature_models[feat] = lr

    future_X = pd.DataFrame({feat: feature_models[feat].predict(future_years.reshape(-1, 1)) for feat in features})

    scaler = StandardScaler()
    X_historical_scaled = scaler.fit_transform(data[features])
    future_X_scaled = scaler.transform(future_X)

    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_historical_scaled, data[target_col])

    y_pred = lasso.predict(future_X_scaled)

    forecast_df = pd.DataFrame({'Год': future_years, f'Прогноз {target_col}': np.round(y_pred, 4)})
    print("\n📊 Прогноз:\n")
    print(forecast_df.to_string(index=False))

if __name__ == "__main__":
    main()
