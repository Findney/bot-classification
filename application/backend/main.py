import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
import pandas as pd
# mmebuat instance FastAPI
app = FastAPI()

# mendefinisikan schema input 
class UserInput(BaseModel):
    ### lengkapi dengan atribut-atribut yang dibutuhkan
    NAME: str
    GENDER: str
    EMAIL_ID: EmailStr
    IS_GLOGIN: bool
    FOLLOWER_COUNT: int
    FOLLOWING_COUNT: int
    DATASET_COUNT: int
    CODE_COUNT: int
    DISCUSSION_COUNT: int
    AVG_NB_READ_TIME_MIN: float
    REGISTRATION_IPV4: str
    REGISTRATION_LOCATION: str
    TOTAL_VOTES_GAVE_NB: int
    TOTAL_VOTES_GAVE_DS: int
    TOTAL_VOTES_GAVE_DC: int

# Model ini adalah hasil dari pipeline gabungan sehingga tidak perlu lagi preprocessing terpisah
model = joblib.load('../../model/best_model_rf.pkl')    

# endpoint untuk menerima input dan menghasilkan prediksi
@app.post("/predict/", summary="Melakukan klasifikasi apakah suatu user tergolong bot atau bukan")
async def predict(user_input: UserInput):
    # Ubah input menjadi format yang sesuai (pandas DataFrame)
    input_dict = user_input.dict()
    data = pd.DataFrame([{
        'NAME': input_dict['NAME'],
        'GENDER': input_dict['GENDER'],
        'EMAIL_ID': input_dict['EMAIL_ID'],
        'IS_GLOGIN': input_dict['IS_GLOGIN'],
        'FOLLOWER_COUNT': input_dict['FOLLOWER_COUNT'],
        'FOLLOWING_COUNT': input_dict['FOLLOWING_COUNT'],
        'DATASET_COUNT': input_dict['DATASET_COUNT'],
        'CODE_COUNT': input_dict['CODE_COUNT'],
        'DISCUSSION_COUNT': input_dict['DISCUSSION_COUNT'],
        'AVG_NB_READ_TIME_MIN': input_dict['AVG_NB_READ_TIME_MIN'],
        'REGISTRATION_IPV4': input_dict['REGISTRATION_IPV4'],
        'REGISTRATION_LOCATION': input_dict['REGISTRATION_LOCATION'],
        'TOTAL_VOTES_GAVE_NB': input_dict['TOTAL_VOTES_GAVE_NB'],
        'TOTAL_VOTES_GAVE_DS': input_dict['TOTAL_VOTES_GAVE_DS'],
        'TOTAL_VOTES_GAVE_DC': input_dict['TOTAL_VOTES_GAVE_DC']
    }])  ## lengkapi dengan data yang akan diproses

    # Drop kolom yang tidak digunakan dalam training
    data = data.drop(columns=['NAME', 'EMAIL_ID', 'REGISTRATION_IPV4', 'REGISTRATION_LOCATION'])

    # Prediksi
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]  # Probabilitas bot (Kelas 1)
    
    return {
        "prediction": int(prediction[0]),
        "bot_probability": float(probability[0]) * 100  # Konversi ke persentase
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)