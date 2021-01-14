import torch as pt

model = pt.load('model.pt')

model = model.eval().cpu()

script_model = pt.jit.script(model, pt.rand(1,3,224,224))

script_model.save("model.pt")