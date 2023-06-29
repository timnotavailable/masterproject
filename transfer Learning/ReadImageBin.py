"""
Definition of reading image.bin data class

Read a single 'image.bin' file
images all rescaled to [0,1]
"""

import numpy as np

      
class ImageBin:
    """ read a image.bin   """
    def __init__(self,file_name,transform=True,*args, **kwargs):
        self.transform=transform
        self.Rchannel=self._read_data(file_name)['Rchannel']
        self.pm1=self._read_data(file_name)['pm1']
        self.pm2=self._read_data(file_name)['pm2']
        self.pm3=self._read_data(file_name)['pm3']
        self.apm1=self._read_data(file_name)['apm1']
        self.apm2=self._read_data(file_name)['apm2']
        self.polarmaps=self._read_data(file_name)['polarmaps']
        self.datadict=self._read_data(file_name)
       
        
        
     
    @staticmethod
    def _read_data(file_name):
        """ 
        Read the data from a given file/ filename
        :file_name : a file object or a file directory.
        :return: 
              Rchannel:
                size:(832,1024) 
                datatype: unsigned int8(uint8)
                Red channel of initialized intepretation of different axis(short, long..... ),several rows 
                
              pm1:
                size:(15,36) 
                datatype: float32
                stress polarmap , transformed from polarmap to a square 
              pm2:
                size:(15,36) 
                datatype: float32
                rest polarmap , transformed from polarmap to a square               
              pm3: pm1-pm2
                size:(15,36) 
                datatype: float32
                delta polarmap , difference of stress and rest
              apm1: 
                size:(15,36) 
                datatype: float32
                area map of  pm1  stress              
              apm2: 
                size:(15,36) 
                datatype: float32
                area map of  pm1  stress                              
              polarmaps: 
                size:(3,256,256) 
                datatype: unsigned int8(uint8)
                Polarmaps, first channel 0: stress polarmaps , second channel 1: rest polarmaps, third channel 2: stress-rest, note that                     polarmaps should be round but here it is set into a 256*256 arrays.
          
          1 unsigned int= 2 byte
          1 float  32  =  4 byte
          
          
          """
        data_dict=None
        
        ## read  image.bin and reshape to different shapes
        Rchannel_byte=1024 * 832*2
        pm1_byte=15*36*4
        pm2_byte=15*36*4
        pm3_byte=15*36*4
        apm1_byte=15*36*4
        apm2_byte=15*36*4
        
        file = open(file_name, "rb")

        Rchannel = np.fromfile(file, np.uint8, 1024 * 832)
        pm1 = np.fromfile(file, np.float32, 15 * 36)
        pm2 = np.fromfile(file, np.float32, 15 * 36)
        pm3 = np.fromfile(file, np.float32, 15 * 36)
        apm1 = np.fromfile(file, np.float32, 15 * 36)
        apm2 = np.fromfile(file, np.float32, 15 * 36)
        polarmaps=np.fromfile(file, np.uint8, 3*256*256)

#         polarmaps=np.reshape(polarmaps,(3,256,256))

        
      ###reshape
        Rchannel=np.reshape(Rchannel,(832,1024))                                  
        pm1 = np.reshape(pm1,(15,36))
        pm2 = np.reshape(pm2,(15,36))
        pm3 = np.reshape(pm3,(15,36))
        apm1 = np.reshape(apm1,(15,36))
        apm2 = np.reshape(apm2,(15,36))
        polarmaps=np.reshape(polarmaps,(3,256,256))
       
       ### Transform
        Rchannel=rescale_transform(Rchannel)
        pm1=rescale_transform(pm1)
        pm2=rescale_transform(pm2)
        pm3=rescale_transform(pm3)
        apm1=rescale_transform(apm1)
        apm2=rescale_transform(apm2)
        for i in range(3) :
            polarmaps[i,:,:]=rescale_transform(polarmaps[i,:,:])        
        
        
        data_dict= { 
            'Rchannel':Rchannel,
            'pm1':pm1,
            'pm2':pm2,
            'pm3':pm3,
            'apm1':apm1,
            'apm2':apm2,
            'polarmaps':polarmaps
                 }
        
        return data_dict
    
    
    
    def flatten(self) :
        """ 
        only flatten pm1, pm2,pm3, 
        """
        self.pm1=np.ravel(self.pm1)
        self.pm2=np.ravel(self.pm2)
        self.pm3=np.ravel(self.pm3)
        
        return
    
    def combine_3(self):
        arr_3=np.stack((self.pm1,self.pm2,self.pm3),axis=0)
        
        
        return arr_3
    
    def combine_4(self):
        arr_4=np.stack((self.pm1,self.pm2,self.apm1,self.apm2),axis=0)
        
        
        return arr_4   
    
    def combine_5(self):
        arr_5=np.stack((self.pm1,self.pm2,self.pm3,self.apm1,self.apm2),axis=0)
        
        return arr_5
    
    def add_gaus_noise(self,mean=0,sigma=0.05):

      
        noise_pm1=np.random.normal(mean,sigma,size=(15, 36))
        self.pm1+=noise_pm1
      
        noise_pm2=np.random.normal(mean,sigma,size=(15, 36))
        self.pm2+=noise_pm2
        
        noise_pm3=np.random.normal(mean,sigma,size=(15, 36))
        self.pm3+=noise_pm3
        
#         print("added Gaussian noise mean is {},sigma is {}".format(mean,sigma))
        

        
        
        
        
        
    
class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]  ##y1
        self.max = out_range[1]  ##y2
        self._data_min = in_range[0]  ##x1
        self._data_max = in_range[1]  ##x2

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        ########################################################################
        slope_k=(self.max-self.min)/(self._data_max-self._data_min)
        bias=self.min-(self.max-self.min)*self._data_min/(self._data_max-self._data_min)
        images=slope_k*images+bias
        pass
               #
        ########################################################################
        return images

    
    
def rescale_transform(image):
    image=(image-image.min()) / (image.max()-image.min())
    return image
    
        
        
        
        
        
        
        
        
        
        
        
        