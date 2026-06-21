
<img width="1890" height="905" alt="image" src="https://github.com/user-attachments/assets/561391a9-8475-4ffd-aab0-d4bb51868bdd" />

<img width="1882" height="820" alt="image" src="https://github.com/user-attachments/assets/18bd0970-26eb-4ed8-a769-6f0c49258f38" />

<img width="1877" height="777" alt="image" src="https://github.com/user-attachments/assets/5e9a4bdb-5cb5-4734-b4c4-28b8d9b0d3b0" />

# Career Compass 🧭

A web-based career recommendation system that suggests suitable career 
paths based on a user's interests, skills and personality — built with 
Flask and cosine similarity matching.

## How It Works

The user rates 19 attributes (interests, skills, personality) on a 
scale of 1–5 using interactive sliders. These ratings are converted 
into a vector and compared against a database of 167 career profiles 
using cosine similarity — the careers with the highest similarity 
scores are recommended as the best matches.

## Features

- 19 interest and skill sliders across three categories
- Cosine similarity matching against 167 career profiles
- Top 5 career recommendations with match percentage and description
- Form remembers your inputs after submission
- Progress bar tracking how many sliders you've adjusted
- Reset button to start over
- Responsive design

## Project Structure

```
career-recommendation-system/
    data/
        career_data_v2.csv      # 167 career profiles with 19 features
    src/
        app.py                  # Flask backend and recommendation logic
        career.ipynb            # Model development and similarity testing
        templates/
            index.html          # Frontend UI
    README.md
    requirements.txt
```

## Setup and Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the application:**
```bash
python src/app.py
```

Open your browser and go to `http://localhost:5000`

## Tech Stack

- Python
- Flask
- Pandas
- Scikit-learn (cosine similarity)
- HTML, CSS, JavaScript

## Possible Improvements

- **Better dataset** — the current career profiles are manually assigned 
  scores. A dataset built from real survey data of professionals in each 
  field would give significantly more accurate recommendations.

- **More careers** — expanding beyond 167 careers, especially adding 
  emerging fields like prompt engineering, AI ethics, climate tech, 
  and creator economy roles.

- **User accounts** — allow users to save their profile and track how 
  their interests change over time.

- **Career detail pages** — clicking on a recommended career could open 
  a page with more information: typical salary, required education, 
  day-to-day responsibilities, and growth outlook.

- **Machine learning model** — replace cosine similarity with a trained 
  classification model using real career survey data for more 
  personalized and accurate recommendations.

- **Feedback loop** — let users rate how accurate their recommendations 
  were, and use that feedback to improve the system over time.

- **Multiple languages** — since this tool could be useful for students 
  in Nepal specifically, adding Nepali language support would make it 
  more accessible.

## Author

Dristi Shakya
shakyadristi2@gmail.com
[LinkedIn](https://linkedin.com/in/dristi-shakya-439908374) |
[GitHub](https://github.com/dristi2365)
