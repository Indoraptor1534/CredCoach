CredCoach AI: An AI-Powered Loan Eligibility Checker & Remediation Coach
(This project is a prototype and is not financial advice.)

The Problem
Traditional automated underwriters often give little explanation to why loans are being rejected. This leaves borrowers confused as to why exactly their application was rejected (whether it is due to a low CIBIL score, lack of liquid assets in their bank or aggressive loan terms compared to their income). Hence, they require an intelligent second opinion to coach them on how to improve their application.

What We Built
CredCoach AI takes an applicant’s financial profiles across 9 factors and returns either an approval verification or a detailed rejection verdict accompanied by:
Weak Point Analysis: Highlights exactly which metrics dragged down their score (e.g., low CIBIL score or insufficient bank asset reserves.).
AI Rectification Roadmap: An interactive list of actionable steps to improve their eligibility.
What if Simulator: Allowing users to tweak their income, savings or loan term and watch their approval odds update live.
What We Added
While standard models only output a binary pass/fail decision, we transformed our model into an active coaching assistant. We added:
Personal Coaching: The system translates model outputs into plain-language financial guidance rather than technical error codes.
What-If Simulator: Users can experiment with financial adjustments to see what changes will get their loan approved.

How It Works
Data: The system collects 9 structured attributes: number of dependents, education level, annual income, self-employment status, loan term,loan amount, CIBIL score, residential assets value and bank assets value.
Model & Backend: The backend is a model that analyzes the eligibility of the applicant using the dataset it has trained from.Built with PyTorch and pandas,it ensures almost instant answers and boasts a 95% to 96% accuracy rate
Frontend Integration: Built with React and Tailwind CSS, featuring smooth transitions, loading states, and fallback mock logic for seamless testing.
Results & Baseline Comparison
Majority class baseline is approximately 63%
A logistic regression model we tried gave an accuracy of 93%
Our current neural network model gives an accuracy of 95%-96%  on 909 applicants

Where It Fails
The dataset we used did not possess data beyond a certain limit.For example in the dataset we only had loan terms from 2 to 20 years,so if the model is to be given beyond those,the model refuses to predict
Problems in our dataset)also led to the model valuing cibil score much more than everything else,with loan money and term coming in second and third,leading to the others to have a negligible value in the decision making process(probably because the dataset seems like it is synthetic)

What We Have To Do Next
Integrate LLM powered conversational coaching so users can ask customized questions about their rectification plan
Find a more accurate and detailed dataset to increase the actual accuracy of the predictions

Sources
https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset for the dataset
Uvicorn, and PyTorch ,scikit-learning and pandas for the model
Bolt.new, React, Tailwind CSS, and Lucide React.
AI coding assistants used for debugging and coding assistance.
