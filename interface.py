import streamlit as st
import pandas as pd
import joblib

# Загрузка обученной модели и списка признаков
rf_model = joblib.load('random_forest_model.pkl')
# Список признаков после обучения
feature_names = joblib.load('feature_names.pkl')

# Заголовок приложения
st.title("Total Price Prediction")

# Ввод данных пользователем
st.sidebar.header("Input Features")


# Функция для ввода данных пользователем
def user_input_features():
    branch = st.sidebar.selectbox("Branch", ["A", "B", "C"])
    city = st.sidebar.selectbox("City", ["Yangon", "Naypyitaw", "Mandalay"])
    customer_type = st.sidebar.selectbox("Customer Type", ["Member", "Normal"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    product_line = st.sidebar.selectbox(
        "Product Line",
        ["Health and beauty", "Electronic accessories", "Home and lifestyle",
         "Sports and travel", "Food and beverages", "Fashion accessories"]
    )
    unit_price = st.sidebar.slider("Unit Price", 10.0, 100.0, 50.0)
    quantity = st.sidebar.slider("Quantity", 1, 10, 5)
    tax = unit_price * quantity * 0.05

    # Формируем DataFrame из пользовательских данных
    data = {
        "Unit price": unit_price,
        "Quantity": quantity,
        "Tax 5%": tax,
        "Branch_B": 1 if branch == "B" else 0,
        "Branch_C": 1 if branch == "C" else 0,
        "City_Naypyitaw": 1 if city == "Naypyitaw" else 0,
        "City_Yangon": 1 if city == "Yangon" else 0,
        "Customer type_Normal": 1 if customer_type == "Normal" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Product line_Electronic accessories":
            1 if product_line == "Electronic accessories" else 0,
        "Product line_Fashion accessories":
            1 if product_line == "Fashion accessories" else 0,
        "Product line_Food and beverages":
            1 if product_line == "Food and beverages" else 0,
        "Product line_Home and lifestyle":
            1 if product_line == "Home and lifestyle" else 0,
        "Product line_Sports and travel":
            1 if product_line == "Sports and travel" else 0,
    }
    features = pd.DataFrame([data])
    return features


# Получаем данные от пользователя
input_data = user_input_features()


# Добавляем отсутствующие признаки (если есть) и упорядочиваем их
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Упорядочиваем признаки в соответствии с обучением модели
input_data = input_data[feature_names]

# Предсказание
prediction = rf_model.predict(input_data)

# Вывод результата
st.subheader("Predicted result:")
st.write(f"**Predicted Total:** {prediction[0]:.2f}")
