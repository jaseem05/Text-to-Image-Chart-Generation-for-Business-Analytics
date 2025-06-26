# 📊 Text-to-Business Chart Generator

An AI-powered web application that converts natural language business queries into insightful, interactive data visualizations. 
Users can upload CSV files and describe the chart they want using plain English—our tool detects the best chart type and generates it instantly using Plotly or Matplotlib.

---

## 🚀 Features

- 🧠 NLP-driven query interpretation
- 📁 Upload CSV data files
- 📊 Auto-detect chart type from text query
- 📈 Generates Bar, Line, Pie, Scatter, and Histogram charts
- ⚡ Uses Plotly for interactive visuals; Matplotlib as fallback
- 🌐 Built using Flask for a lightweight web interface

---

## 🛠 Technologies Used

- *Python 3*
- *Flask* – Web framework
- *Pandas* – Data manipulation
- *Plotly & Matplotlib* – Data visualization
- *HTML/CSS* – Basic front-end templating
- *UUID, OS, JSON* – File and system operations

---

## 🖥 How It Works

1. Upload a .csv file containing your business data.
2. Enter a simple text query like:
   - "Show a bar chart of sales by region"
   - "Create a line chart of revenue over time"
3. The system:
   - Parses the query to detect chart type and relevant columns.
   - Generates and displays the chart.
   - Optionally saves the chart as an HTML file or image.

---

pip install -r requirements.txt
python app.py 
