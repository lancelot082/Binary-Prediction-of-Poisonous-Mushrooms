# Binary-Prediction-of-Poisonous-Mushrooms
![image](https://github.com/user-attachments/assets/c0bbecb9-e518-40e9-9c78-81d50b40f257)

## Overview
Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.
Your Goal: The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.

______

Добро пожаловать на серию игровых площадок Kaggle 2024! Мы планируем продолжать в духе предыдущих игровых площадок, предоставляя интересные и доступные наборы данных для нашего сообщества, чтобы оно могло практиковать свои навыки машинного обучения, и ожидаем, что каждый месяц будет проводиться соревнование.
Ваша цель: Цель этого конкурса - предсказать, является ли гриб съедобным или ядовитым, основываясь на его физических характеристиках.

______

## Score
accuracy: 0.98
_____

## optuna
<pre>
  ```
  # Определите вашу функцию для оптимизации гиперпараметров
def objective(trial):
    # Разделение данных на тренировочные и тестовые
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Создание пайплайна для числовых данных
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Создание пайплайна для категориальных данных
    categorical_pipeline = Pipeline([
        ('imputer1', SimpleImputer(strategy='most_frequent')),
        ('encoding', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])


    # Трансформатор данных
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Определение модели с гиперпараметрами для оптимизации
    model = CatBoostClassifier(
        iterations=trial.suggest_int('iterations', 100, 500),
        depth=trial.suggest_int('depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 10),
        random_state=42,
        verbose = 0
    )

    # Пайплайн с препроцессором и моделью
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Обучение модели
    model_pipeline.fit(X_train, y_train)

    # Оценка качества модели на валидационном наборе данных
    y_pred = model_pipeline.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

    return accuracy

# Определение гиперпараметров и оптимизация с использованием Optuna
study = optuna.create_study(direction='maximize')  # maximize для максимизации accuracy
study.optimize(objective, n_trials=20)

# Вывод лучших гиперпараметров
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
for key, value in trial.params.items():
    print(f'  {key}: {value}')
  ```
</pre>


## pipeline
<pre>
  ```
  # Разделение данных на тренировочные и тестовые
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание пайплайна для числовых данных
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

# Создание пайплайна для категориальных данных
categorical_pipeline = Pipeline([
    ('imputer1', SimpleImputer(strategy='most_frequent')),
    ('encoding', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Трансформатор данных
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Заданные параметры
params = {
    'iterations': 275,
    'depth': 10,
    'learning_rate': 0.09855752946871991,
    'l2_leaf_reg': 5.8220284047377735
}

# Определение модели с гиперпараметрами
model = CatBoostClassifier(**params, random_state=42, verbose=0)


# Пайплайн с препроцессором и моделью
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Обучение модели
model_pipeline.fit(X_train, y_train)

# Оценка качества модели на валидационном наборе данных
y_pred = model_pipeline.predict(X_valid)


accuracy = accuracy_score(y_valid, y_pred)


print(accuracy)
  ```
</pre>
