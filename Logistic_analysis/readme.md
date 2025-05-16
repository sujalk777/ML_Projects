# Workflow for Predicting Consignment Pricing

This project aims to predict the consignment pricing using classical machine learning techniques. Below is a detailed workflow and execution guide.

## 1. Data Preparation

- **Dataset**: The dataset contains multiple features, including shipment details, product characteristics, and costs.
- **Cleaning**:
  - Convert numerical columns like `Weight (Kilograms)` and `Freight Cost (USD)` to proper data types.
  - Drop rows with missing values in critical columns such as `Weight (Kilograms)`, `Freight Cost (USD)`, and `Shipment Mode`.
- **Feature Selection**: Relevant features used for modeling include:
  - `Country`
  - `Shipment Mode`
  - `Weight (Kilograms)`
  - `Line Item Quantity`
  - `Line Item Value`
  - `Vendor INCO Term`

## 2. Model Building

- Split the data into training and testing sets (80-20 split).
- Use a **preprocessing pipeline** to:
  - Scale numerical features (`Weight (Kilograms)`, `Line Item Quantity`, `Line Item Value`) using `StandardScaler`.
  - Encode categorical features (`Country`, `Shipment Mode`, `Vendor INCO Term`) using `OneHotEncoder`.

### Machine Learning Models

- **Linear Regression**: A baseline model to establish initial performance.
- **Random Forest Regressor**: An ensemble model to improve prediction accuracy.

## 3. Model Evaluation

- Train the models and evaluate them using **Mean Squared Error (MSE)**.
- Save trained pipelines for reuse using `joblib`.

### Results

The MSE for both models is calculated and printed.

## 4. Visualization

- Generate scatter plots comparing true values and predicted values for both models.
- Include a diagonal reference line to assess model performance visually.

### Code Snippet for Plotting

```python
# Plot Predictions
plt.figure(figsize=(14, 6))
for i, (model_name, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(1, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{model_name} Predictions")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
plt.tight_layout()
plt.show()
```

## 5. Execution

1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python main.py
   ```

4. View results and plots generated in the terminal or as saved files.

## 6. File Structure

- **main.py**: The main script for data preprocessing, model training, evaluation, and visualization.
- **requirements.txt**: Lists the required Python libraries.
- **README.md**: Documentation and workflow.
- **models/**: Directory to save trained pipelines.
- **data/**: Directory for the dataset.

## 7. Key Libraries Used

- `pandas`: Data manipulation.
- `scikit-learn`: Machine learning and preprocessing.
- `matplotlib`: Visualization.
- `joblib`: Model saving and loading.

## 8. Future Work

- Experiment with additional machine learning algorithms (e.g., Gradient Boosting, XGBoost).
- Perform hyperparameter tuning to further optimize models.
- Deploy the model using a web framework (e.g., Flask or FastAPI).

## 9. Contributions

Feel free to contribute to this project by submitting issues or pull requests.

