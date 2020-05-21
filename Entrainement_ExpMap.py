# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 08:41:27 2020

@author: Lucien

https://medium.com/@layog/a-comprehensive-overview-of-pytorch-7f70b061963f
https://medium.com/@ml_kid/what-is-transform-and-transform-normalize-lesson-4-neural-networks-in-pytorch-ca97842336bd
"""

import torch
import xlrd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import transforms as transforms
import torch.nn.functional as F

from Classes_ExpMap_avec_FC import Generator, Discriminator, quat2expmap




# Liste de variables____________________________________________________________________________________________________


# Liste d'adresses
AdresseBaseDeDonnees = 'D:\\Projet\\DVGAN_Student\\bvh2npz\\data npz\\Quaternion\\walk_run_uneven_jump.npz' #changer quaternion -> ExpMap?
AdresseFichierAction = 'D:\\Projet\\data\\action.xlsx'

list_dloss = []
list_gloss = []


NombreFramesParGeste = 32
batch = 32

class opt:
    n_epochs = 3000
    batch_size = 32
    lr = 0.0001
    n_critic = 10
    b1 = 0.9
    b2 = 0.999
    # Loss weight for gradient penalty
    lambda_gp = 10

cuda = True if torch.cuda.is_available() else False



#_Liste de fonctions____________________________________________________________________________________________________


def Definition_H_txt_global():
    shape = []
    wb = xlrd.open_workbook(AdresseFichierAction)
    sheet = wb.sheet_by_index(0)
    shape.append(sheet.nrows)
    shape.append(sheet.ncols)

    '''
    Déclaration des variables
    '''
    dictionnaire = {}  # Contient dans l'ordre du document la description du mouvement. Ex.: 'walk'
    dicovecteur = {}   # Contient dans l'ordre du document les représentations par one-hot vector (de 256)
    H_txt = {}         # Contient l'attribution des vecteurs pour chaque nonm : Clé = nom & Valeur = vecteur
    lecteur1 = 0       # Ce sont les compteurs pour parcourir le fichier action
    lecteur2 = 0

    '''
    Remplissage de dictionnaire
    '''
    for col in range(shape[1]):
        if (col % 2 == 0):
            for row in range(shape[0]):
                if (len(sheet.cell_value(row, col)) != 0):
                    dictionnaire[lecteur1] = sheet.cell_value(row, col + 1)
                    lecteur1 += 1

    '''
    Définition des vecteurs
    '''

    size = 256

    # Liste de tailles de type de données
    nbrDeJump = 30
    nbrDeWalkOnUnevenTerrain = 33
    nbrDeRun = 34

    for t in range(lecteur1):
        v = np.zeros(size)
        v[3] = 1
        dicovecteur[t] = v

    for t in range(lecteur1 - nbrDeJump):
        v = np.zeros(size)
        v[2] = 1
        dicovecteur[t] = v

    for t in range(lecteur1 - nbrDeJump - nbrDeWalkOnUnevenTerrain):
        v = np.zeros(size)
        v[1] = 1
        dicovecteur[t] = v

    for t in range(lecteur1 - nbrDeJump - nbrDeWalkOnUnevenTerrain - nbrDeRun):
        v = np.zeros(size)
        v[0] = 1
        dicovecteur[t] = v

    '''
    Remplissage de H_txt
    '''
    for col2 in range(shape[1]):
        if (col2 % 2 == 0):  # Files
            for row2 in range(shape[0]):
                if (len(sheet.cell_value(row2, col2)) != 0):
                    temp = sheet.cell_value(row2, col2)
                    H_txt[temp] = dicovecteur[lecteur2]
                    lecteur2 += 1

    return dictionnaire, np.array(list(H_txt.items()))



def new_load(filename):                 # Nouvelle fonction np.load() car la traditionnelle produit des erreurs
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true
    data = np.load(filename)
    # restore np.load for future normal usage
    np.load = np_load_old

    return data



def compute_gradient_penalty(D, real_samples, fake_samples, htexte):
    """Calculates the gradient penalty loss for WGAN"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad = True

    '''
    print('real sample', real_samples.shape)
    print('fake sample', fake_samples.shape)
    print('d_interpolate',  D(interpolates, htexte).shape)
    print('len(real_samples)', len(real_samples))
    '''

    d_interpolates = D(interpolates, htexte)  # .reshape(len(real_samples), 1)
    if cuda:
        d_interpolates = d_interpolates.cuda()
    # print('dinterpolates', d_interpolates.shape)

    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        # allow_unused=True,     #test
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if cuda:
        gradient_penalty = gradient_penalty.cuda()

    return gradient_penalty



#_Classe utile au dataloader____________________________________________________________________________________________


class CustomTensorDataset(Dataset):

    def __init__(self, filename):
        data = new_load(filename)
        #data = np.load(filename)
        data = data['clips']                            # Les frames se trouvent dans les clips
        liste = []                                      # Création d'une liste dans laquelle on va stocker les frames
        mvt = []
        Nframes = 0
        TailleOriginale = 0
        TailleFinale = 0

        print('\nTéléchargement de la base de données à partir d un dossier de', len(data),'fichiers.')

        # Paragraphe pour le retrait des 'run'
        (dictionnaire,_) = Definition_H_txt_global()
        (_,H_text_global) = Definition_H_txt_global()
        H_text_global = H_text_global[:,1]

        '''
        idxs = []
        for j in range(len(data)):
            if(dictionnaire[j]=='run' or dictionnaire[j]=='run/jog' or dictionnaire[j] == 'Run'):
                idxs.append(j)
                TailleOriginale += data[j].shape[0]
        data = np.delete(data, idxs)
        H_text_global = np.delete(H_text_global, idxs)
        print('Run removed, data with shape : ', data.shape[0])
        '''

        for i in range(len(data)):
            geste120Hz = data[i][1:data[i].shape[0]]
            geste8Hz = geste120Hz[::15]
            TailleOriginale += data[i].shape[0]

            if (geste8Hz.shape[0] >= NombreFramesParGeste):
                gesteScalp32 = geste8Hz[0:NombreFramesParGeste]
                #print(gesteScalp32.shape, '\n', gesteScalp32.dtype)
                liste.append(gesteScalp32)
                Nframes += geste8Hz.shape[0]
                TailleFinale += gesteScalp32.shape[0]
                #print(geste8Hz.shape[0])
                mvt.append(H_text_global[i])
    
            if (geste8Hz.shape[0] < NombreFramesParGeste):
                gesteAllonge = np.zeros([NombreFramesParGeste, 31, 4])         # Devra être modifié en dimensions exponential maps
                for a in range(NombreFramesParGeste):
                    if (a < geste8Hz.shape[0]):
                        gesteAllonge[a] = geste8Hz[a]
                    else:
                        gesteAllonge[a] = geste8Hz[geste8Hz.shape[0]-1]
                          
                liste.append(gesteAllonge)
                Nframes += geste8Hz.shape[0]
                TailleFinale += gesteAllonge.shape[0]
                mvt.append(H_text_global[i])
        

        #CONVERSION EN EXPMAP (en se servant de la fonction quat2expmap fournie par monsieur Maio)
        liste = np.asarray(liste)
        liste = np.apply_along_axis(quat2expmap, 3, liste)  #cette fonction de np permet d'appliquer une fonction sur 
                                                            #une dimension spécifique d'un array
        NombreGestes = len(liste)                                                    
        #print('shape liste', liste.shape)
        #print('nombreGestes', NombreGestes)
        #Normalisation des données______________________________________________________________________________
        #ici on flatten
        
        liste = liste.reshape(liste.shape[0]*NombreFramesParGeste, -1)
        
        #print('liste après flatten', liste.shape)
        
        #ici on normalise
        mean = liste.mean(axis=0)        
        std = liste.std(axis=0)    
        
        #print('std:', std)
        #print('mean :', mean)    
        
        liste = (liste - mean)/(std + 1e-6)  
        
        #print('liste après normalisation', liste.shape)  
        
        #on déflatten
        liste = liste.reshape(NombreGestes, NombreFramesParGeste, 93) 
        #print('liste après unflatten: ', liste.shape)  #[196, 32, 93]
 
        

        self.frames = liste
        self.NbrFrames = TailleFinale
        self.mouvement = np.asarray(mvt)
        
        self.moyenne = mean
        self.variance = std

        print('Base de longueur', Nframes, 'chargée\t(taille originale de', TailleOriginale, 'frames) !')
        print('Base redimensionnée en fichiers de 32 frames. Taille finale de :', TailleFinale,'frames pour un total de',len(self.frames),'mouvements ! \n \n')
        print('Gestes normalisés avec succès !')
        
    def __getitem__(self, index):
        return torch.tensor((self.frames[index])), self.mouvement[index]


    def __len__(self):
        return len(self.frames)



#_Main__________________________________________________________________________________________________________________


# Initialize generator and discriminator
generator = Generator()
generator = generator.double()

discriminator = Discriminator()
discriminator = discriminator.double()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()



# Dataloader
dataset = CustomTensorDataset(AdresseBaseDeDonnees)

NombreTotalDeFrames = dataset.NbrFrames
NombreTotalDeMouvements = len(dataset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)



# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




# ----------
#  Training
# ----------
compteur = 0
batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (batchGeste, batchH_txt) in enumerate(dataloader):
       
        #batchGeste = batchGeste.reshape(batchGeste.shape[0], NombreFramesParGeste, -1) #j'ai reshape dans le dataloader
        #print('batch shape:', batchGeste.shape)
        
        print('epoch: ', epoch)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = []
        for k in range(0, 7):
            if cuda:
                z.append((torch.randn(len(batchGeste), 2**k, 256).double()).cuda())
            else:
                z.append(torch.randn(len(batchGeste), 2**k, 256).double())

        if cuda:
            batchGeste = batchGeste.cuda()
            batchH_txt = batchH_txt.cuda()

        fake_mvt = generator(z, batchH_txt)
        #print('fake_mvt', fake_mvt.grad_fn)

        real_validity = discriminator(batchGeste, batchH_txt)
        fake_validity = discriminator(fake_mvt, batchH_txt)
        if cuda:
            real_validity = real_validity.cuda()
            fake_validity = fake_validity.cuda()


        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, batchGeste.data, fake_mvt.data, batchH_txt)


        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()


        #print('epoch: ', epoch,  'compteur : ', compteur, 'modulo : ', compteur%opt.n_critic, )
        # Train the generator every n_critic steps
        if compteur % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            #print('je génère coucou')
            # Generate a batch of movements
            fake_mvt = generator(z, batchH_txt)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake movements
            fake_validity = discriminator(fake_mvt, batchH_txt)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

            """print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )"""

            batches_done += opt.n_critic

            list_gloss.append(g_loss.item())         
            list_dloss.append(d_loss.item())

        compteur +=1

#_Sauvegarde de l'entrainement__________________________________________________________________________________________


print('Done! Currenlty saving models...')
torch.save(generator.state_dict(), 'generator_TEST_NDB.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

#np.save('Moyenne_Gestes', dataset.moyenne)
#np.save('Std_Gestes', dataset.variance)


#_Génération d'un graphique pour l'observation de l'évolution des loss__________________________________________________
print('Training Done!')
plt.plot(list_gloss, c='r', label='Generator')
plt.plot(list_dloss, c='b', label='Discriminator')
plt.title("Evolution of gloss and dloss according to the number of epochs\n[n_Epoch: %d] [Batch_size: %d] [n_critic: %d]" % (opt.n_epochs, opt.batch_size, opt.n_critic))
plt.legend(loc='upper right');
plt.xlabel('Number of passes through the generator')
plt.ylabel('Loss values')
plt.show()
print('Last dloss',list_dloss[len(list_gloss)-1],'\t\tLast gloss', list_gloss[len(list_gloss)-1],'\n')



'''

plt.plot(list_gloss, c='r')
plt.plot(list_dloss, c='b')
plt.title("generator in red - discriminator in blue, [n_Epoch: %d] [Batch_size: %d] [n_critic: %d]" % (opt.n_epochs, opt.batch_size, opt.n_critic))
plt.show()

print('Last dloss',list_dloss[len(list_gloss)-1],'\t\tLast gloss',
list_gloss[len(list_gloss)-1])
'''