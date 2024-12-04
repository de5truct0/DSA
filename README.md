# Data Analysis Pipeline with GPT-4 Integration

This project provides an end-to-end data analysis pipeline that integrates data preprocessing, analysis, visualization and API-based insights using GPT-4. The script supports various data formats and simplifies the data science workflow.

---

## Features

- **Data Loading**: Supports `.csv`, `.json`, `.xlsx`, and `.parquet` file formats.
- **Data Preprocessing**: Handles missing values, duplicate removal, categorical encoding, and numeric scaling.
- **Data Analysis**: Provides basic statistics, correlation analysis, and visualizations.
- **Statistical Analysis**: Performs normality and correlation tests.
- **Machine Learning**: Supports classification, regression, and clustering models.
- **Visualization**: Generates interactive plots using Plotly.
- **GPT-4 Integration**: Sends data summaries and prompts to GPT-4 for insights.

---

## Requirements

### Python Packages

Install required libraries using:

```bash
pip install -r requirements.txt
```

**Key Libraries:**

- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning and preprocessing
- `scipy`: Statistical analysis
- `matplotlib`, `seaborn`, `plotly`: Data visualization
- `aiohttp`: Asynchronous API communication
- `python-dotenv`: Environment variable management

### Environment Variables

Create a `.env` file in the root directory with the following:

```
API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your OpenAI API key.

---

## How to Run

### 1. Run the Script

Execute the script in your terminal:

```bash
python datasc.py
```

### 2. User Prompts

- **Data File Path**: Provide the path to your dataset (e.g., `data.csv`).
- **Target Column Name**: Specify the target column for ML models or leave blank if none.
- **Analysis Prompt**: Enter a custom prompt for GPT-4 to analyze the data.

---

## Components

### 1. **Data Loading**

The `DataLoader` class validates file formats and loads data into a pandas DataFrame.

Supported formats:

- `.csv`
- `.json`
- `.xlsx`
- `.parquet`

### 2. **Data Preprocessing**

The `DataPreprocessor` class handles:

- Missing values imputation
- Duplicate removal
- Categorical encoding (using `LabelEncoder`)
- Numeric feature scaling (using `StandardScaler`)

### 3. **Data Analysis**

The `DataAnalyzer` class provides:

- Basic statistics (`describe()`, data types)
- Correlation matrix
- Missing value counts

### 4. **Data Visualization**

The `DataVisualizer` class generates:

- Distribution histograms
- Correlation heatmaps
- Box plots for outlier detection

### 5. **Statistical Analysis**

The `StatisticalAnalyzer` class includes:

- Normality tests
- Pearson correlation tests

### 6. **Machine Learning**

The `MachineLearning` class supports:

- Classification (`RandomForestClassifier`, `LogisticRegression`)
- Regression (`LinearRegression`, `RandomForestRegressor`)
- Clustering (`KMeans`)

### 7. **GPT-4 Integration**

The `APIHandler` class sends data summaries and custom prompts to GPT-4 and retrieves insights.

---

### Adding More Models

To include additional ML models, modify the `MachineLearning` class:

```python
from sklearn.svm import SVC
self.models['svm_classifier'] = SVC()
```

### Modifying GPT-4 Prompts

Edit the `messages` structure in the `APIHandler.send_data_async` method to tailor the prompt format.
