import numpy as np
import cv2
import pickle

# Словарь классов
target_dict = {1: 'notsmoking', 0: 'smoking'}

# Загрузка модели
with open('mymodel.pkl', 'rb') as file:
    model = pickle.load(file)

cap = cv2.VideoCapture(0)  # Открываем камеру


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Изменяем размер для модели
    image = cv2.resize(frame, (250, 250))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1, 250, 250, 1)
    image = image / 255

    # Предсказание модели
    y_pred = model.predict(image)
    print(y_pred)
    text = target_dict[np.argmax(y_pred)]  # Определяем класс

    # Добавляем текст на экран
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображаем видео
    cv2.imshow('Camera', frame)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



