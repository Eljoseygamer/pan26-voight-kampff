import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from utils import load_input, load_train, write_predictions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def main():
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    train_path = os.path.join(os.path.dirname(__file__), 'data', 'train.jsonl')
    if not os.path.exists(train_path):
        train_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.jsonl')
    print('Loading training data...')
    X_train, y_train = load_train(train_path)
    print('Training TF-IDF + SVM...')
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2, sublinear_tf=True)),
        ('clf', CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=2000), cv=3))
    ])
    pipe.fit(X_train, y_train)
    print('Loading test data...')
    items = load_input(input_file)
    ids = [x[0] for x in items]
    texts = [x[1] for x in items]
    scores = pipe.predict_proba(texts)[:, 1].tolist()
    write_predictions(output_dir, ids, scores)


if __name__ == '__main__':
    main()
