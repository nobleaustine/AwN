from unet import UNetModel_v1preview1
from torchview import draw_graph

m = UNetModel_v1preview1()

model_graph = draw_graph(m, input_size=(1, 3, 224, 224) , device='meta')
model_graph.visual_graph