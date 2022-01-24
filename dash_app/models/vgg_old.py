import collections, torch, torchvision, numpy
import models.vgg as vgg

# Return a version of vgg11 where the layers are given their research names
def vgg11(*args, tiny=False, **kwargs):
    model = vgg.vgg11(*args, tiny=tiny, **kwargs)
    model.features = torch.nn.Sequential(collections.OrderedDict(zip([
        'conv1_1', 'relu1_1',
        'pool1',
        'conv2_1', 'relu2_1',
        'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'pool5'],
        model.features)))

    model.classifier = torch.nn.Sequential(collections.OrderedDict(zip([
        'fc6', 'relu6',
        'drop6',
        'fc7', 'relu7',
        'drop7',
        'fc8a'],
        model.classifier)))

    return model


def vgg11_bn(*args, tiny=False, **kwargs):
    model = vgg.vgg11_bn(*args, tiny=tiny, **kwargs)
    model.features = torch.nn.Sequential(collections.OrderedDict(zip([
        'conv1_1', 'relu1_1', 'bn1_1',
        'pool1',
        'conv2_1', 'relu2_1', 'bn2_1',
        'pool2',
        'conv3_1', 'relu3_1', 'bn3_1',
        'conv3_2', 'relu3_2', 'bn3_2',
        'pool3',
        'conv4_1', 'relu4_1', 'bn4_1',
        'conv4_2', 'relu4_2', 'bn4_2',
        'pool4',
        'conv5_1', 'relu5_1', 'bn5_1',
        'conv5_2', 'relu5_2', 'bn5_2',
        'pool5'],
        model.features)))

    model.classifier = torch.nn.Sequential(collections.OrderedDict(zip([
        'fc6', 'relu6',
        'drop6',
        'fc7', 'relu7',
        'drop7',
        'fc8a'],
        model.classifier)))

    return model


def vgg16(*args, tiny=False, **kwargs):
    model = vgg.vgg16(*args, tiny=tiny, **kwargs)
    
    model.features = torch.nn.Sequential(collections.OrderedDict(zip([
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2',
        'pool1',
        'conv2_1', 'relu2_1',
        'conv2_2', 'relu2_2',
        'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3',
        'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3',
        'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3',
        'pool5'],
        model.features)))

    model.classifier = torch.nn.Sequential(collections.OrderedDict(zip([
        'fc6', 'relu6',
        'drop6',
        'fc7', 'relu7',
        'drop7',
        'fc8a'],
        model.classifier)))
    
    return model


def vgg16_bn(*args, tiny=False, **kwargs):
    model = vgg.vgg16_bn(*args, tiny=tiny, **kwargs)
    
    model.features = torch.nn.Sequential(collections.OrderedDict(zip([
        'conv1_1', 'relu1_1', 'bn1_1', 
        'conv1_2', 'relu1_2', 'bn1_2', 
        'pool1',
        'conv2_1', 'relu2_1', 'bn2_1', 
        'conv2_2', 'relu2_2', 'bn2_2', 
        'pool2',
        'conv3_1', 'relu3_1', 'bn3_1', 
        'conv3_2', 'relu3_2', 'bn3_2', 
        'conv3_3', 'relu3_3', 'bn3_3', 
        'pool3',
        'conv4_1', 'relu4_1', 'bn4_1', 
        'conv4_2', 'relu4_2', 'bn4_2', 
        'conv4_3', 'relu4_3', 'bn4_3', 
        'pool4',
        'conv5_1', 'relu5_1', 'bn5_1', 
        'conv5_2', 'relu5_2', 'bn5_2', 
        'conv5_3', 'relu5_3', 'bn5_3', 
        'pool5'],
        model.features)))

    model.classifier = torch.nn.Sequential(collections.OrderedDict(zip([
        'fc6', 'relu6',
        'drop6',
        'fc7', 'relu7',
        'drop7',
        'fc8a'],
        model.classifier)))
    
    return model