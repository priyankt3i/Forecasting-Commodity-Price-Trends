# Analyzing and Forecasting Commodity Price Trends Using Data-Driven Models

## Overview
This repository contains the research and analysis conducted in the study *Analyzing and Forecasting Commodity Price Trends Using Data-Driven Models*. The study focuses on predicting the prices of five key commodities—Gold, Silver, Copper, Palladium, and Platinum—using advanced statistical, machine learning, and deep learning techniques. The analysis includes data exploration, model development, and actionable insights tailored for financial and industrial stakeholders.

---

## Key Features
- **Commodities Covered**: Gold, Silver, Copper, Palladium, and Platinum.
- **Time Period**: 2000–2024.
- **Methodologies Used**:
  - **Statistical Models**: ARIMA and Prophet for time-series forecasting.
  - **Machine Learning Models**: Random Forest for capturing non-linear relationships.
  - **Deep Learning Models**: Long Short-Term Memory (LSTM) networks for handling sequential data and long-term dependencies.
  - **Hybrid Approach**: Combines ARIMA, Random Forest, Prophet, and LSTM for enhanced predictive accuracy.
- **Key Outputs**:
  - Price trends, volatility analysis, and correlation insights.
  - Predictive models for commodity price forecasting.
  - Practical recommendations for stakeholders.

---

## File Structure
- **`data/`**: Historical commodity price data and cleaned datasets.
- **`notebooks/`**: Jupyter notebooks for data preprocessing, EDA, and model development.
  - `eda.ipynb`: Exploratory data analysis.
  - `arima_model.ipynb`: ARIMA modeling.
  - `prophet_model.ipynb`: Prophet modeling.
  - `random_forest_model.ipynb`: Random Forest modeling.
  - `lstm_model.ipynb`: LSTM development and training.
  - `hybrid_model.ipynb`: Combining models for hybrid predictions.
- **`models/`**: Serialized models for ARIMA, Prophet, Random Forest, LSTM, and hybrid predictions.
- **`visualizations/`**: Graphs and plots showcasing trends, correlations, and model results.
- **`results/`**: Evaluation metrics and analysis reports for the models.
- **`docs/`**: Research documentation, references, and detailed methodology.

---

## Dependencies
This project is implemented in Python and requires the following libraries:
- `pandas`, `numpy`: Data preprocessing and analysis.
- `matplotlib`, `seaborn`: Data visualization.
- `statsmodels`: ARIMA modeling.
- `scikit-learn`: Random Forest implementation.
- `prophet`: Time-series forecasting with Prophet.
- `tensorflow` or `pytorch`: LSTM implementation.
- `jupyter`: For interactive analysis and notebooks.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Running the Project
1. **Data Preprocessing**:
   - Load the historical data from the `data/` folder.
   - Use the `notebooks/data_cleaning.ipynb` notebook to preprocess the data.

2. **Exploratory Data Analysis (EDA)**:
   - Run the `notebooks/eda.ipynb` notebook to analyze trends, correlations, and volatility.

3. **Model Training**:
   - Train ARIMA models using `notebooks/arima_model.ipynb`.
   - Train Prophet models using `notebooks/prophet_model.ipynb`.
   - Train Random Forest models using `notebooks/random_forest_model.ipynb`.
   - Train LSTM models using `notebooks/lstm_model.ipynb`.
   - Combine predictions in the hybrid approach using `notebooks/hybrid_model.ipynb`.

4. **Evaluation**:
   - Review performance metrics in the `results/` folder.

5. **Visualizations**:
   - View key visualizations in the `visualizations/` folder.

---

## Key Findings
- **Correlations**: Gold and Silver exhibit a strong positive correlation (~0.85), reflecting shared economic drivers.
- **Volatility**: Palladium is the most volatile commodity, influenced by supply-demand imbalances.
- **Model Performance**:
  - ARIMA and Prophet effectively capture linear trends and seasonality.
  - Random Forest excels in identifying non-linear relationships.
  - LSTM captures long-term dependencies in sequential data.
  - The hybrid model outperforms individual methods for high-volatility commodities.

---

## Future Enhancements
- Integrating additional external macroeconomic and geopolitical factors.
- Expanding analysis to energy and agricultural commodities.
- Refining LSTM models using advanced architectures like Transformers.
- Developing real-time dashboards for adaptive forecasting.

---

## Contributors
This study was conducted by Rishav Mondal. Please contact for queries or collaborations.

---

## References
The methodology and analysis are supported by academic literature, industry reports, and open-source tools. Key references include:
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/) for LSTM implementation.
- [Kaggle Datasets](https://www.kaggle.com/)

For a complete list of references, see the `docs/references.pdf`.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

--- 

Feel free to adjust further to align with your specific goals!
