from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cubic_eos import CubicEOS, get_species

app = FastAPI(title="Real Gas Calculator API")

# Allow your HTML website to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalculationInput(BaseModel):
    species: str
    eos: str
    temperature: float
    pressure: float
    phase: str = "auto"


@app.get("/")
def home():
    return {"message": "Real Gas Calculator API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/calculate")
def calculate(data: CalculationInput):
    try:
        # Retrieve the selected gas and equation of state
        gas = get_species(data.species)
        model = CubicEOS(gas, data.eos)

        # Calculate thermodynamic properties
        result = model.properties(
            data.temperature,
            data.pressure,
            data.phase
        )

        state = result.state

        return {
            "species": state.species,
            "eos": state.eos,
            "temperature": state.T,
            "pressure": state.P,
            "phase": state.phase,
            "Z": state.Z,
            "volume": state.V,
            "enthalpy": result.H,
            "entropy": result.S,
            "residual_enthalpy": state.HR,
            "residual_entropy": state.SR,
            "fugacity_coefficient": state.phi,
            "fugacity": state.f,
            "roots": state.roots,
            "warning": result.warning,
            "note": state.note
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
