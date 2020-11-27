from .builder import SEIRDBuilder
from .model import SEIRDModel

SIMPLE_MODEL = SEIRDBuilder(0.25, D_E=1.0).with_stream().build()

ASYMMETRIC_MODEL = SEIRDBuilder(0.25, D_E=1.0).with_stream(K_D=5).build()