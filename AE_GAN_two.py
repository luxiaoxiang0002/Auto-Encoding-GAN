import argparse
import os
import random

import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=15, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

file_name = "R15_images"
os.makedirs(file_name, exist_ok=True)
root = os.getcwd()
generator_path = root + "/" + file_name + "/" + "generator_model"
isExists = os.path.exists(generator_path)
if not isExists:
    os.mkdir(generator_path)

discriminator_path = root + "/" + file_name + "/"+ "/discriminator_model"
isExists = os.path.exists(discriminator_path)
if not isExists:
    os.mkdir(discriminator_path)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = torch.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return y_cat

def randomcolor():
    # 随机产生不同的颜色
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def Normalization(dataset, min_value, max_value):
    # 将数据集归一化到[min_value, max_value]
    dim_min = np.min(dataset)
    dim_max = np.max(dataset)
    if (abs(dim_max - dim_min) <= 0.000001):
        dataset = dataset
    else:
        dataset = ((dataset - dim_min) / (dim_max - dim_min)) * (max_value - min_value) + min_value

    return dataset

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.n_classes, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, 0.8),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.BatchNorm1d(2, 0.8),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        gen_input = torch.cat((z, labels), -1)
        fake_out = self.model(gen_input)
        return fake_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(64, 0.8),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(128, 0.8),
            nn.Linear(128, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        # The height and width of downsampled image
        ds_size = 512

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(ds_size, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(ds_size, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(ds_size, opt.code_dim))

    def forward(self, img):
        feature_out = self.model(img)
        validity = self.adv_layer(feature_out)
        label = self.aux_layer(feature_out)

        return validity, label, feature_out

def ShowImage(dataset, total_gen_data, each_gen_num, epoch):
    root = os.getcwd()
    test_path = root + "/"+ file_name +"/" + "test_result"
    isExists = os.path.exists(test_path)
    if not isExists:
        os.mkdir(test_path)

    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    dataset = dataset.cpu().detach().numpy()
    plt.scatter(dataset[:, 0], dataset[:, 1], color='b', marker='.')
    for g in range(opt.n_classes):
        gen_data = total_gen_data[g*each_gen_num:(g+1)*each_gen_num]
        color = randomcolor()
        for l in range(gen_data.shape[0]):
            plt.text(gen_data[l, 0], gen_data[l, 1], str(g), color=color,
                     fontdict={'weight': 'bold', 'size': 9}, )

    plt.savefig(test_path + '/gan_%d.jpg' % epoch)
    plt.close()

def classifier_result(dataset, y, cluster_num, epoch):

    root = os.getcwd()
    tsne_path = root + "/"+ file_name +"/" + "classifier_result"
    isExists = os.path.exists(tsne_path)
    if not isExists:
        os.mkdir(tsne_path)

    '''嵌入空间可视化'''
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    j = 0
    while (j <= cluster_num):
        color = randomcolor()
        for i in range(dataset.shape[0]):
            if (y[i] == j ):
                plt.text(dataset[i, 0], dataset[i, 1], str(int(y[i])), color=color, fontdict={'weight': 'bold', 'size': 9},
                         alpha=0.4)
        j = j + 1
    plt.savefig(tsne_path + '/' + "image_%d.jpeg" % epoch)
    plt.close()

def feature_space_kmeans_cluster(gen_feature_dataset, real_feature_dataset, k, each_gen_num):

    gen_feature_dataset = gen_feature_dataset.cpu().detach().numpy()
    real_feature_dataset = real_feature_dataset.cpu().detach().numpy()

    gen_cluster_model = KMeans(n_clusters=k)
    gen_kmeans_pred = gen_cluster_model.fit_predict(gen_feature_dataset)
    gen_centers = gen_cluster_model.cluster_centers_

    real_cluster_model = KMeans(n_clusters=k)
    real_kmeans_pred = real_cluster_model.fit_predict(real_feature_dataset)
    real_centers = real_cluster_model.cluster_centers_

    clusters = np.unique(gen_kmeans_pred)

    #centers_matrix = euclidean_dist(real_centers, gen_centers)
    centers_matrix = F.cosine_similarity(torch.tensor(real_centers).unsqueeze(1), torch.tensor(gen_centers).unsqueeze(0), dim=2)

    adjust_real_kmeans_pred = np.zeros(real_kmeans_pred.shape[0], dtype=np.int32) - 1
    for i in range(centers_matrix.shape[0]):
        index = torch.argmax(centers_matrix[i])
        adjust_real_kmeans_pred[np.where(real_kmeans_pred == i)] = index

    real_kmeans_pred = adjust_real_kmeans_pred

    gen_y_pred = np.zeros(gen_kmeans_pred.shape[0], dtype=np.int32)
    real_y_pred = np.zeros(real_kmeans_pred.shape[0], dtype=np.int32)
    for i in range(clusters.shape[0]):
        y = gen_kmeans_pred[i * each_gen_num:(i + 1) * each_gen_num]
        cluster_label = np.argmax(np.bincount(y))
        gen_y_pred[i * each_gen_num:(i + 1) * each_gen_num] = i
        real_y_pred[np.where(real_kmeans_pred == cluster_label)] = i

    return gen_centers, real_y_pred

# Loss functions
adversarial_loss = torch.nn.BCELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# Configure data loader
root = os.getcwd()
dataset_file = root + "/" + "dataset/R15.txt"
X = np.loadtxt(dataset_file)
X = X[:, 0:2]
dataset = Normalization(X, -0.8, 0.8)

dataset = torch.tensor(dataset)
dataset = dataset.float()
dataset = dataset.to(device)


dataloader = torch.utils.data.DataLoader(
    dataset=dataset,  # torch TensorDataset format
    batch_size=opt.batch_size,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)




# ----------
#  Training
# ----------
alpha = 0.0
for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        each_gen_num = 10  # 每个G的输入数据量

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        real = Variable(FloatTensor(each_gen_num * opt.n_classes, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(each_gen_num * opt.n_classes, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (each_gen_num * opt.n_classes, opt.latent_dim))))

        mutil_gen_target = torch.empty((0, 1), dtype=torch.long).cuda()
        mutil_gen_target.to(device)
        for j in range(opt.n_classes):
            gen_target = torch.zeros((each_gen_num, 1), dtype=torch.long)
            gen_target = (gen_target + j).cuda()
            gen_target.to(device)
            mutil_gen_target = torch.cat((mutil_gen_target, gen_target), dim=0)

        mutil_gen_target = mutil_gen_target.view(mutil_gen_target.shape[0])  # 将标签向量变成一行
        mutil_one_hot_code = to_categorical(mutil_gen_target, opt.n_classes)  # 转化成one-hot编码,作为隐编码输入
        mutil_one_hot_code = Variable(mutil_one_hot_code)
        mutil_one_hot_code = mutil_one_hot_code.to(device)

        # label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)

        # Generate a batch of images
        gen_imgs = generator(z, mutil_one_hot_code)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, real)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # gen + real Loss
        # ------------------

        optimizer_info.zero_grad()

        gen_imgs = generator(z, mutil_one_hot_code)
        _, gen_pred_label, gen_feature = discriminator(gen_imgs)
        _, real_pred_label, real_feature = discriminator(real_imgs)

        gen_cluster_loss = categorical_loss(gen_pred_label, mutil_gen_target)

        gen_feature = F.normalize(gen_feature)
        real_feature = F.normalize(real_feature)
        centers, real_cluster_target = feature_space_kmeans_cluster(gen_feature, real_feature, opt.n_classes,
                                                                    each_gen_num)

        real_cluster_target = torch.tensor(real_cluster_target, dtype=torch.long)
        real_cluster_target = Variable(real_cluster_target).to(device)

        real_cluster_loss = categorical_loss(real_pred_label, real_cluster_target)

        gc_loss = gen_cluster_loss + alpha * real_cluster_loss

        gc_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [gc loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), gc_loss.item())
        )
        if epoch <= int(opt.n_epochs * 0.7):
            alpha = 0.001
        else:
            alpha = 0.1
        batches_done = epoch
        if batches_done % opt.sample_interval == 0:
            each_gen_num = 20
            z = Variable(FloatTensor(np.random.normal(0, 1, (each_gen_num * opt.n_classes, opt.latent_dim)))).to(device)
            mutil_gen_target = torch.empty((0, 1), dtype=torch.long).cuda()
            mutil_gen_target.to(device)
            for j in range(opt.n_classes):
                gen_target = torch.zeros((each_gen_num, 1), dtype=torch.long)
                gen_target = (gen_target + j).cuda()
                gen_target.to(device)
                mutil_gen_target = torch.cat((mutil_gen_target, gen_target), dim=0)

            mutil_gen_target = mutil_gen_target.view(mutil_gen_target.shape[0])  # 将标签向量变成一行
            mutil_one_hot_code = to_categorical(mutil_gen_target, opt.n_classes)  # 转化成one-hot编码,作为隐编码输入
            mutil_one_hot_code = mutil_one_hot_code.to(device)

            #generator.eval()
            gen_imgs = generator(z, mutil_one_hot_code)
            ShowImage(dataset, gen_imgs, each_gen_num, epoch)

            #discriminator.eval()
            c_out = discriminator(dataset)[1]
            y_pred = torch.max(c_out, 1)[1]
            print("分类器预测标签", torch.unique(y_pred))

            classifier_result(dataset, y_pred, opt.n_classes, epoch)

            discriminator_model_dir = discriminator_path + "/discriminator_%d.pth" % (epoch)
            state = {'model': discriminator.state_dict()}
            torch.save(state, discriminator_model_dir)

            generator_model_dir = generator_path + "/generator_%d.pth" % (epoch)
            state = {'model':generator.state_dict()}
            torch.save(state, generator_model_dir)
