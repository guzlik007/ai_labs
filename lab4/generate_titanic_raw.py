import os
from datetime import datetime, date

import numpy as np
import pandas as pd

BASE_RAW_DIR = 'data/raw'
N_ROWS = 100

FEATURE_COLS = [
    'PassengerId',
    'Survived',
    'Pclass',
    'Name',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Ticket',
    'Fare',
    'Cabin',
    'Embarked'
]

FIRST_NAMES = ['John', 'William', 'James', 'George', 'Charles', 'Thomas', 'Henry', 'Arthur']
LAST_NAMES = ['Smith', 'Brown', 'Williams', 'Taylor', 'Davies', 'Evans', 'Thomas', 'Roberts']
TITLES = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.']


def generate_name():
    title = np.random.choice(TITLES)
    first = np.random.choice(FIRST_NAMES)
    last = np.random.choice(LAST_NAMES)
    return f'{last}, {title} {first}'


def generate_ticket():
    prefix = np.random.choice(['PC', 'A/5', 'STON/O2', 'C', 'SOTON'])
    number = np.random.randint(1000, 999999)
    return f'{prefix} {number}'


def generate_cabin():
    if np.random.rand() < 0.2:
        return np.nan
    letter = np.random.choice(list('ABCDE'))
    number = np.random.randint(1, 250)
    return f'{letter}{number}'


def inject_missing(df):
    df = df.copy()
    n = len(df)

    for col, frac in [
        ('Age', 0.12),
        ('Fare', 0.08),
        ('Embarked', 0.05),
        ('Cabin', 0.05),
        ('Sex', 0.02),
    ]:
        k = int(n * frac)
        if k == 0:
            continue
        idx = np.random.choice(df.index, size=k, replace=False)
        df.loc[idx, col] = np.nan

    return df


def generate_full_titanic(n):
    np.random.seed()

    df = pd.DataFrame(
        {
            'PassengerId': np.arange(1, n + 1),
            'Survived': np.random.choice([0, 1], size=n),
            'Pclass': np.random.choice([1, 2, 3], size=n),
            'Name': [generate_name() for _ in range(n)],
            'Sex': np.random.choice(['male', 'female'], size=n),
            'Age': np.random.uniform(1, 80, size=n),
            'SibSp': np.random.randint(0, 6, size=n),
            'Parch': np.random.randint(0, 6, size=n),
            'Ticket': [generate_ticket() for _ in range(n)],
            'Fare': np.random.uniform(5, 300, size=n),
            'Cabin': [generate_cabin() for _ in range(n)],
            'Embarked': np.random.choice(['S', 'C', 'Q'], size=n),
        }
    )

    df = inject_missing(df)
    return df[FEATURE_COLS]


def save_csv(df, for_date=None):
    if for_date is None:
        for_date = date.today()

    ds = for_date.isoformat()
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')

    folder = os.path.join(BASE_RAW_DIR, ds)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f'raw_{ts}.csv')
    df.to_csv(file_path, index=False)
    return file_path


def main():
    df = generate_full_titanic(N_ROWS)
    path = save_csv(df)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
