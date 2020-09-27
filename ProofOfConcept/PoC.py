#Proof Of Concept
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

from datetime import datetime
from keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.utils import plot_model

class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 1)
        #self.latent_dim = (self.seq_length, 1)
        self.latent_dim = 1000
        self.disc_loss = []
        self.gen_loss =[]
        
        #optimizer = RMSprop(lr=0.00005)
        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates note sequences
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_seq)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        #Discriminator_Architecture_Version 20191227_01
        model = Sequential()
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        plot_model(model, to_file= current_time + "_discriminator.png", show_layer_names=True, show_shapes=True)
        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        return Model(seq, validity)
      
    def build_generator(self):
        #Generator_Architecture_Version 20191227_01
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        model.summary()
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        plot_model(model, to_file= current_time + "_generator.png", show_layer_names=True, show_shapes=True)
        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)

        return Model(noise, seq)

    def make_testing_and_training_file(self, SyscallLength=15):
        RawData = np.genfromtxt('VX_Results_clean_' + str(SyscallLength) + '_calls_per_sample.csv',dtype=int,delimiter=',')
        #RawData is a 2D array with an index for the y and the data set for x
        TrainingData =  np.zeros(shape=(1280,SyscallLength),dtype=int)
        TrainingIndices = np.zeros(shape=1600,dtype=int)
        TrainingIndices = np.random.choice(RawData.shape[0], 1600, replace=False)
        temp = np.zeros(shape=SyscallLength,dtype=int)
        for x in range (0,1279):
            temp = RawData[TrainingIndices[x],:]
            np.transpose(temp)
            np.copyto(TrainingData[x,:], temp, casting='unsafe')
        
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        np.savetxt('training_data.csv',TrainingData,fmt='%i',delimiter=',')
        np.savetxt('training_data_' + current_time + '.csv',TrainingData,fmt='%i',delimiter=',')
        
        TestingData =  np.zeros(shape=(320,SyscallLength),dtype=int)
        for y in range (0,319):
            temp = RawData[TrainingIndices[y+1279],:]
            np.transpose(temp)
            np.copyto(TestingData[y,:], temp, casting='unsafe')
        
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        np.savetxt('testing_data_' + current_time +'.csv',TestingData,fmt='%i',delimiter=',')
        np.savetxt('testing_data.csv',TestingData,fmt='%i',delimiter=',')
        
        

    def train(self, epochs=1000, sequence_length=15,samples=10):
    
    #Pull all the data into memory as raw. This is better than constantly accessing the disk
        RawData = np.genfromtxt('training_data.csv', delimiter=',')
        #samples = 10
    
        real = np.ones((samples, 1))
        fake = np.zeros((samples, 1))

        sample_batch = np.zeros(shape=(samples,sequence_length,1), dtype=int)
                
        # Training the model
        for epoch in range(epochs):

            for sample in range(samples):
                for timestep in range(sequence_length-1):
                    sample_batch[sample,timestep,0] = RawData[sample,timestep]

            # So assuming that we have to have -1 to 1. 
            # We have a range of 0 to 547. This gives us 548 descrete values.
            # The half way mark would be 274
            # normalized = samples - 274. This gives a range of -274 to 273 
            # normalized = x / 274 ; and keeping the polairty sign 
            
            normalized = ((sample_batch - 274)/274)

            # Training the discriminator
            #the random.normal indicates a gaussian distribution
            noise = np.random.normal(0, 1, (samples, self.latent_dim))

            # Generate a batch of new system calls
            gen_seqs = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(normalized, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  Training the Generator
            noise = np.random.normal(0, 1, (samples, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            #noise = np.random.normal(0, 1, (samples, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            #noise = np.random.normal(0, 1, (samples, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)


            # Print the progress and save into loss lists
            #if epoch % sample_interval == 0:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            self.disc_loss.append(d_loss[0])
            self.gen_loss.append(g_loss)
        
        self.plot_loss()


    def IsMalware(self, sequence_length=15):

        reader2 = csv.reader(open("fake_malware.csv"))
        reader = csv.reader(open("testing_data.csv"))
        reader1 = csv.reader(open("Legit_Results_" + str(sequence_length) + ".csv"))
        f = open("predictions.csv", "w", newline='')
        writer = csv.writer(f)
        for row in reader:
            writer.writerow(row)
        for row in reader1:
            writer.writerow(row)
        for row in reader2:
            writer.writerow(row)
        f.close()

        sample_list = np.genfromtxt('predictions.csv', delimiter=',')
        number_of_samples = len(sample_list)
        #candidate_sample=np.zeros(shape=(number_of_samples,sequence_length,1), dtype=int)
        candidate_sample=np.zeros(shape=(1,sequence_length,1),dtype=int)
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        output = open("results_" + current_time + ".csv", "w", newline='')
        OutputWriter = csv.writer(output)

        for i in range(number_of_samples):
            #for j in range(sequence_length-1):
            np.transpose(sample_list)
            candidate_sample[0,:,0] = sample_list[i]
            classified = self.discriminator.predict(candidate_sample)
            #print("Classification is :", classified)
            OutputWriter.writerow((classified*1000))

        output.close()

        #np.savetxt('final_results.csv',finalresults,delimiter=',')
        

    def GenerateMalwareSysCalls(self,number_of_malware_samples=320,SyscallLength=15):
        
        #We would just look at a batch of them to show what could be done next
        #This could be interesting in that the calls would be defined but the rest
        #Could be left to the imagination
        FakeMalware =  np.zeros(shape=(320,SyscallLength),dtype=int)
        SyscallList = np.zeros(shape=(SyscallLength,1))
        print("GenerateMalwareSysCalls\n")
        for i in range (0,(number_of_malware_samples)):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            X = self.generator.predict(noise)
            SyscallList = ((X * 274) + 274)
            np.transpose(SyscallList)
            SyscallList.astype(int,casting='unsafe')
            syscalls = np.squeeze(SyscallList,axis=2)
            np.copyto(FakeMalware[i,:], syscalls, casting='unsafe')

        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        np.savetxt('fake_malware.csv',FakeMalware,fmt='%f',delimiter=',')
        np.savetxt('fake_malware_' + current_time + '.csv',FakeMalware,fmt='%f',delimiter=',')
        
        
    def plot_loss(self):
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(current_time + '_GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

if __name__ == '__main__':
  gan = GAN(rows=25)    
  gan.make_testing_and_training_file(SyscallLength=25)
  gan.train(epochs=125, sequence_length=25,samples=60)
  gan.GenerateMalwareSysCalls(number_of_malware_samples=320,SyscallLength=25)
  gan.IsMalware(sequence_length=25)