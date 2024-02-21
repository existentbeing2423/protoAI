from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)