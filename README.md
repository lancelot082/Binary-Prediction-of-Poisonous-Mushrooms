# Binary-Prediction-of-Poisonous-Mushrooms
![image](https://github.com/user-attachments/assets/c0bbecb9-e518-40e9-9c78-81d50b40f257)

## Overview
The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.

______

 Цель этого конкурса - предсказать, является ли гриб съедобным или ядовитым, основываясь на его физических характеристиках.

______

## Данные

- `train.csv`: тренировочная выборка с метками классов
- `test.csv`: тестовая выборка без меток
- `sample_submission.csv`: шаблон отправки

## Этапы работы

### 1. Исследование и очистка данных

- Обнаружены пропуски и невалидные значения в категориальных признаках — заменены на `NaN`, если символ не является однобуквенным.
- Удалены признаки с избыточным количеством пропусков (>80%) или без полезной информации (`id`, `veil-type`, `spore-print-color` и др.).
- Категориальные и числовые признаки разделены для дальнейшей обработки.

### 2. Обработка признаков

- Пропущенные значения:
  - Числовые — заполнены средним значением;
  - Категориальные — модой.
- Для категориальных данных используется **OneHotEncoding** (через `ColumnTransformer`).
- Предобработка завернута в `Pipeline` для удобства и повторного использования.

### 3. Модель

Используется `CatBoostClassifier` с предварительно подобранными параметрами:

```python
params = {
    'iterations': 275,
    'depth': 10,
    'learning_rate': 0.0985,
    'l2_leaf_reg': 5.82
}
```

Модель обучается внутри `Pipeline`, без необходимости вручную кодировать категориальные переменные.

### 4. Валидация

- Разделение на train/validation через `train_test_split`
- Оценка точности модели (`accuracy_score`)
- Получена стабильная точность на валидации

### 5. Предсказание и сабмит

- Прогноз сделан на `test.csv`
- Результат сохранен в `submission.csv` в формате Kaggle

```python
submission = pd.DataFrame({
    'id' : test['id'],
    'class' : y_pred_test
})
submission.to_csv('submission.csv', index=False)
```
