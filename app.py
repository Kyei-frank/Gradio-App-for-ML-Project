# Importing required Libraries
from IPython.utils.py3compat import encode
import gradio as gr
import numpy as np
import pandas as pd
import pickle


# Loading Machine Learning Objects
def load_saved_objets(filepath='ML_items'):
    "Function to load saved objects"

    with open(filepath, 'rb') as file:
        loaded_object = pickle.load(file)
    
    return loaded_object

# Instantiating ML_items
loaded_object = load_saved_objets()
pipeline_of_my_app = loaded_object["pipeline"]
num_cols = loaded_object['numeric_columns']
cat_cols = loaded_object['categorical_columns']
encoder_categories = loaded_object["encoder_categories"]

# Main function to collect the inputs process them and outpuT the predicition
def predict_churn(
    TotalCharges,
    MonthlyCharges,
    tenure, 
    StreamingTV,
    PaperlessBilling,
    DeviceProtection,
    TechSupport,
    InternetService,
    OnlineSecurity,
    StreamingMovies,
    PaymentMethod,
    Dependents,
    Parter,
    tenure_group,
    OnlineBackup,
    gender,
    SeniorCitizen,
    MultipleLines,
    Contract,
    PhoneService,
):
    
    df = pd.DataFrame(
        [
            [
                TotalCharges,
                MonthlyCharges,
                tenure, 
                StreamingTV,
                PaperlessBilling,
                DeviceProtection,
                TechSupport,
                InternetService,
                OnlineSecurity,
                StreamingMovies,
                PaymentMethod,
                Dependents,
                Parter,
                tenure_group,
                OnlineBackup,
                gender,
                SeniorCitizen,
                MultipleLines,
                Contract,
                PhoneService,
            ]
        ],  
        columns= num_cols + cat_cols,
    ).replace("", np.nan)
    
    df[cat_cols] = df[cat_cols].astype("object")
    
    # Passing data to pipeline to make prediction
    output = pipeline_of_my_app.predict(df)
    
    # Labelling Model output
    if output == 0:
        model_output = "No"
    else:
        model_output = "Yes"

    return model_output
 

# Setting up app interface and data inputs
inputs = []

with gr.Blocks() as demo:
    
    # Setting Titles for App
    gr.Markdown("<h2 style='text-align: center;'> Customer Churn Prediction App </h2> ", unsafe_allow_html=True)
    gr.Markdown("<h6 style='text-align: center;'> (Fill in the details below and click on PREDICT button to make a prediction for Customer Churn) </h6> ", unsafe_allow_html=True)   
    
    with gr.Column(): #main frame 
        
        with gr.Row(): #col 1 : for num features

            for i in num_cols:
                inputs.append(gr.Number(label=f"Input {i} "))
        
        with gr.Row(): #col 2 : for cat features

            for (lab, choices) in zip(cat_cols, encoder_categories):
                inputs.append(gr.inputs.Dropdown(
                choices=choices.tolist(),
                type="value",
                label=f"Select {lab}",
                default=choices.tolist()[0],))
    # Setting up preediction Button
    with gr.Row():
        make_prediction = gr.Button("Predict")
    
    # Setting up prediction output row
    with gr.Row():
        output_prediction = gr.Text(label="Will Customer Churn?")
    make_prediction.click(predict_churn, inputs, output_prediction)

# Launching app
demo.launch(
    share=True,
    # debug=True
)