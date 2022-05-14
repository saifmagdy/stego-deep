# Create your models here.

from email.mime import image
import cv2
import numpy as np 
import types
import base64
import wave
import bitarray
import bitstring
import struct



HORIZ_AXIS = 1
VERT_AXIS  = 0

# Standard quantization table as defined by JPEG
JPEG_STD_LUM_QUANT_TABLE = np.asarray([
                                        [16, 11, 10, 16,  24, 40,   51,  61],
                                        [12, 12, 14, 19,  26, 58,   60,  55],
                                        [14, 13, 16, 24,  40, 57,   69,  56],
                                        [14, 17, 22, 29,  51, 87,   80,  62],
                                        [18, 22, 37, 56,  68, 109, 103,  77],
                                        [24, 36, 55, 64,  81, 104, 113,  92],
                                        [49, 64, 78, 87, 103, 121, 120, 101],
                                        [72, 92, 95, 98, 112, 100, 103,  99]
                                      ],
                                      dtype = np.float32)
# Image container class


    
class YCC_Image(object):
    def __init__(self, cover_image):
        self.height, self.width = cover_image.shape[:2]
        d = DCT()
        self.channels = [
                         d.split_image_into_8x8_blocks(cover_image[:,:,0]),
                         d.split_image_into_8x8_blocks(cover_image[:,:,1]),
                         d.split_image_into_8x8_blocks(cover_image[:,:,2])
                        ]



      
class DCT():
    
    def __init__(self):
        pass

    def extract_encoded_data_from_DCT(self, dct_blocks):
        extracted_data = ""
        for current_dct_block in dct_blocks:
            for i in range(1, len(current_dct_block)):
                curr_coeff = np.int32(current_dct_block[i])
                if (curr_coeff > 1):
                    extracted_data += bitstring.pack('uint:1', np.uint8(current_dct_block[i]) & 0x01)
        return extracted_data

# ============================================================================= #
# ============================================================================= #
# ============================================================================= #
# ============================================================================= #

    def embed_encoded_data_into_DCT(self, encoded_bits, dct_blocks):
        data_complete = False; encoded_bits.pos = 0
        encoded_data_len = bitstring.pack('uint:32', len(encoded_bits))
        converted_blocks = []
        for current_dct_block in dct_blocks:
            for i in range(1, len(current_dct_block)):
                curr_coeff = np.int32(current_dct_block[i])
                if (curr_coeff > 1):
                    curr_coeff = np.uint8(current_dct_block[i])
                    if (encoded_bits.pos == (len(encoded_bits) - 1)): data_complete = True; break
                    pack_coeff = bitstring.pack('uint:8', curr_coeff)
                    if (encoded_data_len.pos <= len(encoded_data_len) - 1): pack_coeff[-1] = encoded_data_len.read(1)
                    else: pack_coeff[-1] = encoded_bits.read(1)
                    # Replace converted coefficient
                    current_dct_block[i] = np.float32(pack_coeff.read('uint:8'))
            converted_blocks.append(current_dct_block)
        
        if not(data_complete): raise ValueError("Data didn't fully embed into cover image!")
        return converted_blocks



#====================================================================================================#
#====================================================================================================#

    def stitch_8x8_blocks_back_together(self, Nc, block_segments):
        '''
        Take the array of 8x8 pixel blocks and put them together by row so the numpy.block() method can sitch it back together
        :param Nc: Number of pixels in the image (length-wise)
        :param block_segments:
        :return:
        '''
        image_rows = []
        temp = []
        for i in range(len(block_segments)):
            if i > 0 and not(i % int(Nc / 8)):
                image_rows.append(temp)
                temp = [block_segments[i]]
            else:
                temp.append(block_segments[i])
        image_rows.append(temp)

        return np.block(image_rows)

    #====================================================================================================#
    #====================================================================================================#

    def split_image_into_8x8_blocks(self, image):
        blocks = []
        for vert_slice in np.vsplit(image, int(image.shape[0] / 8)):
            for horiz_slice in np.hsplit(vert_slice, int(image.shape[1] / 8)):
                blocks.append(horiz_slice)
        return blocks

    

    def zigzag(self, input):
        #initializing the variables
	#----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input.shape[0]
        hmax = input.shape[1]

        #print(vmax ,hmax )

        i = 0

        output = np.zeros(( vmax * hmax))
        #----------------------------------

        while ((v < vmax) and (h < hmax)):

            if ((h + v) % 2) == 0:                 # going up

                if (v == vmin):
                    #print(1)
                    output[i] = input[v, h]        # if we got to the first line

                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1

                    i = i + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    #print(2)
                    output[i] = input[v, h]
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    #print(3)
                    output[i] = input[v, h]
                    v = v - 1
                    h = h + 1
                    i = i + 1


            else:                                    # going down

                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    #print(4)
                    output[i] = input[v, h]
                    h = h + 1
                    i = i + 1

                elif (h == hmin):                  # if we got to the first column
                    #print(5)
                    output[i] = input[v, h]

                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1

                    i = i + 1

                elif ((v < vmax -1) and (h > hmin)):     # all other cases
                    #print(6)
                    output[i] = input[v, h]
                    v = v + 1
                    h = h - 1
                    i = i + 1




            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                #print(7)
                output[i] = input[v, h]
                break

        #print ('v:',v,', h:',h,', i:',i)
        return output


       
    def inverse_zigzag(self, input, vmax, hmax):
        	#print input.shape

	# initializing the variables
	#----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        output = np.zeros((vmax, hmax))

        i = 0
        #----------------------------------

        while ((v < vmax) and (h < hmax)):
            #print ('v:',v,', h:',h,', i:',i)
            if ((h + v) % 2) == 0:                 # going up

                if (v == vmin):
                    #print(1)

                    output[v, h] = input[i]        # if we got to the first line

                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1

                    i = i + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    #print(2)
                    output[v, h] = input[i]
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    #print(3)
                    output[v, h] = input[i]
                    v = v - 1
                    h = h + 1
                    i = i + 1


            else:                                    # going down

                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    #print(4)
                    output[v, h] = input[i]
                    h = h + 1
                    i = i + 1

                elif (h == hmin):                  # if we got to the first column
                    #print(5)
                    output[v, h] = input[i]
                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1

                elif((v < vmax -1) and (h > hmin)):     # all other cases
                    output[v, h] = input[i]
                    v = v + 1
                    h = h - 1
                    i = i + 1




            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                #print(7)
                output[v, h] = input[i]
                break


        return output

    def DCT_encode(self,image_name,data,filename):
        raw_cover_image = cv2.imread(image_name, flags=cv2.IMREAD_COLOR)
        height, width   = raw_cover_image.shape[:2]
        # Force Image Dimensions to be 8x8 compliant
        while(height % 8): height += 1 # Rows
        while(width  % 8): width  += 1 # Cols
        valid_dim = (width, height)
        padded_image    = cv2.resize(raw_cover_image, valid_dim)
        cover_image_f32 = np.float32(padded_image)
        cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

        # Placeholder for holding stego image data
        stego_image = np.empty_like(cover_image_f32)

        for chan_index in range(3):
            # FORWARD DCT STAGE
            dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

            # QUANTIZATION STAGE
            dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]


            # Sort DCT coefficients by frequency
            sorted_coefficients = [self.zigzag(block) for block in dct_quants]
            
            # Embed data in Luminance layer
            if (chan_index == 0):
                # DATA INSERTION STAGE
                secret_data = ""
                for char in data.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
                embedded_dct_blocks   = self.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                desorted_coefficients = [self.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
            else:
                # Reorder coefficients to how they originally were
                desorted_coefficients = [self.inverse_zigzag(block, vmax=8,hmax=8) for block in sorted_coefficients]

            # DEQUANTIZATION STAGE
            dct_dequants = [np.multiply(data, JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]

            # Inverse DCT Stage
            idct_blocks = [cv2.idct(block) for block in dct_dequants]

            # Rebuild full image channel
            stego_image[:,:,chan_index] = np.asarray(self.stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

            # Convert back to RGB (BGR) Colorspace
        stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

        # Clamp Pixel Values to [0 - 255]
        final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))

        # Write stego image
        cv2.imwrite(filename, final_stego_image)

    def extract_text(self,image_name):
        stego_image     = cv2.imread(image_name, flags=cv2.IMREAD_COLOR)
        stego_image_f32 = np.float32(stego_image)
        stego_image_YCC = YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

        # FORWARD DCT STAGE
        dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]  # Only care about Luminance layer

        # QUANTIZATION STAGE
        dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

        # Sort DCT coefficients by frequency
        sorted_coefficients = [self.zigzag(block) for block in dct_quants]

        # DATA EXTRACTION STAGE
        recovered_data = self.extract_encoded_data_from_DCT(sorted_coefficients)

        # Determine length of secret message
        data_len = int(recovered_data.read('uint:32') / 8)

        # Extract secret message from DCT coefficients
        extracted_data = bytes()
        for _ in range(data_len): extracted_data += struct.pack('>B', recovered_data.read('uint:8'))

        # Print secret message back to the user
        print(extracted_data.decode('ascii'))


class Image_LSB():

    def __init__(self):
        pass

    def messageToBinary(self, message):
        if type(message) == str:
            return ''.join([ format(ord(i), "08b") for i in message ])
        elif type(message) == bytes or type(message) == np.ndarray:
            return [ format(i, "08b") for i in message ]
        elif type(message) == int or type(message) == np.uint8:
            return format(message, "08b")
        else:
            raise TypeError("Input type not supported")

    def hideData(self, image, secret_message):
    #Check if the number of bytes to encode is less than the maximum bytes in the image
        n_bytes = image.shape[0] * image.shape[1] * 3 // 8
        n_bytes_double = image.shape[0] * image.shape[1] * 3 * 2 // 8
        secret_message += "#####" # you can use any string as the delimeter
        data_index = 0
    # convert input data to binary format using messageToBinary() fucntion
        binary_secret_msg = self.messageToBinary(secret_message)
        data_len = len(binary_secret_msg) #Find the length of data that needs to be hidden
        if data_len < n_bytes:
            print("will use lsb technique as Maximum bytes to encode:" , n_bytes,data_len)
            for values in image:
                for pixel in values:
                    # convert RGB values to binary format
                    r, g, b = self.messageToBinary(pixel)
                    # modify the least significant bit only if there is still data to store
                    if data_index < data_len:
                        # hide the data into least significant bit of red pixel
                        pixel[0] = int(r[:-1] + binary_secret_msg[data_index], 2)
                        data_index += 1
                    if data_index < data_len:
                # hide the data into least significant bit of green pixel
                        pixel[1] = int(g[:-1] + binary_secret_msg[data_index], 2)
                        data_index += 1
                    if data_index < data_len:
                # hide the data into least significant bit of  blue pixel
                        pixel[2] = int(b[:-1] + binary_secret_msg[data_index], 2)
                        data_index += 1
            # if data is encoded, just break out of the loop
                    if data_index >= data_len:
                        break
        
        
        elif n_bytes_double > data_len > n_bytes:
            for values in image:
                for pixel in values:
                    r, g, b = self.messageToBinary(pixel)
                    if data_index < data_len:
                # hide the data into least significant bit of red pixel
                        pixel[0] = int(r[:-2] + binary_secret_msg[data_index]+ binary_secret_msg[data_index + 1], 2)
                        data_index += 2
                    if data_index < data_len:
                # hide the data into least significant bit of green pixel
                        pixel[1] = int(g[:-2] + binary_secret_msg[data_index]+ binary_secret_msg[data_index + 1], 2)
                        data_index += 2
                    if data_index < data_len:
                # hide the data into least significant bit of  blue pixel
                        pixel[2] = int(b[:-2] + binary_secret_msg[data_index]+ binary_secret_msg[data_index + 1], 2)
                        data_index += 2
            # if data is encoded, just break out of the loop
                    if data_index >= data_len:
                        break
        return image

    def showData(self, image):
        binary_data = ""
        for values in image:
            for pixel in values:
                r, g, b = self.messageToBinary(pixel) #convert the red,green and blue values into binary format
                binary_data += r[-2] #extracting data from the least significant bit of red pixel  
                binary_data += r[-1] #extracting data from the least significant bit of red pixel
                binary_data += g[-2] #extracting data from the least significant bit of red pixel
                binary_data += g[-1] #extracting data from the least significant bit of red pixel
                binary_data += b[-2] #extracting data from the least significant bit of red pixel
                binary_data += b[-1] #extracting data from the least significant bit of red pixel
    # split by 8-bits
        all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
        #convert from bits to characters
        decoded_data = ""
        for byte in all_bytes:
            decoded_data += chr(int(byte, 2))
            if decoded_data[-5:] == "#####": #check if we have reached the delimeter which is "#####"
                break
    #print(decoded_data)
        return decoded_data[:-5]

    def showDataLeast(self, image):
        binary_data = ""
        for values in image:
            for pixel in values:
                r, g, b = self.messageToBinary(pixel) #convert the red,green and blue values into binary format
                binary_data += r[-1] #extracting data from the least significant bit of red pixel
                binary_data += g[-1] #extracting data from the least significant bit of red pixel
                binary_data += b[-1] #extracting data from the least significant bit of red pixel
                # split by 8-bits
        all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
    # convert from bits to characters
        decoded_data = ""
        for byte in all_bytes:
            decoded_data += chr(int(byte, 2))
            if decoded_data[-5:] == "#####": #check if we have reached the delimeter which is "#####"
                break
    #print(decoded_data)
        return decoded_data[:-5]

    def encode_text(self,image_name,data,filename):  
        image = cv2.imread(image_name)  
        if (len(data) == 0): 
            raise ValueError('Data is empty')
        encoded_image = self.hideData(image, data) # call the hideData function to hide the secret message into the selected image
        cv2.imwrite(filename, encoded_image)

    def decode_text(self,image_name):
        image = cv2.imread(image_name)
        text = self.showData(image)
        return text

    def decode_textLeast(self,image_name):
        image = cv2.imread(image_name)
        text = self.showDataLeast(image)
        return text

            

class Audio_LSB():
   
   
   
   def __init__(self,audio_name):

        print(audio_name)
        self.audio_name = audio_name
        self.audio = wave.open(audio_name,mode="rb")
        self.frame_bytes = bytearray(list(self.audio.readframes(self.audio.getnframes())))

   
   def isValid(self, string, framebytes):
      bits = bitarray.bitarray()
      bits.frombytes(string.encode('utf-8'))
      if len(bits) < len(frame_bytes):
         return True
      else:
         return False
   

   def encode(self, string):
      string = string + '#####'
      ba= bitarray.bitarray()
      ba.frombytes(string.encode('utf-8'))
      bits = ba.tolist()

      for i, bit in enumerate(bits):
         self.frame_bytes[i] = (self.frame_bytes[i] & 254) | bit
      frame_modified = bytes(self.frame_bytes)
    
      newAudio =  wave.open('samplelsb.wav', 'wb')
      newAudio.setparams(self.audio.getparams())
      newAudio.writeframes(frame_modified)
    

      newAudio.close()
      self.audio.close()
      
      
   def twoEncode(self, string):
      string = string + '#####'
    
      ba= bitarray.bitarray()
      ba.frombytes(string.encode('utf-8'))
      bits = ba.tolist()
    
    
      j=0
      for i in range(0,(len(bits)//2)):
         tmp = str(int(bits[j])) + str(int(bits[j+1]))
         self.frame_bytes[i] = (self.frame_bytes[i] & 252) | int(tmp,2)
         j+=2
      frame_modified = bytes(self.frame_bytes)
    
      newAudio =  wave.open('twolsb.wav', 'wb')
      newAudio.setparams(self.audio.getparams())
      newAudio.writeframes(frame_modified)

      newAudio.close()
      self.audio.close()
      
      
   def decode(self):
      audio = wave.open(self.audio_name, mode='rb')
      self.frame_bytes = bytearray(list(audio.readframes(audio.getnframes())))
      extracted = [self.frame_bytes[i] & 1 for i in range(len(self.frame_bytes))]
    
      string = bitarray.bitarray(extracted).tobytes().decode('utf-8','ignore')
      self.audio.close()	
      decoded = string.split("#####")[0]
      return decoded
   
   
   def twoDecode(self):
      audio = wave.open("twolsb.wav", mode='rb')
      frame_bytes = bytearray(list(audio.readframes(audio.getnframes())))
      extracted = ['{:02b}'.format(frame_bytes[i] & 3) for i in range(len(frame_bytes))]
      extracted = ''.join(extracted)
      string = bitarray.bitarray(extracted).tobytes().decode('utf-8','ignore')
      audio.close()
      decoded = string.split("#####")[0]
      return decoded
      
class DCT():
    def extract_encoded_data_from_DCT(dct_blocks):
        extracted_data = ""
        for current_dct_block in dct_blocks:
            for i in range(1, len(current_dct_block)):
                curr_coeff = np.int32(current_dct_block[i])
                if (curr_coeff > 1):
                    extracted_data += bitstring.pack('uint:1', np.uint8(current_dct_block[i]) & 0x01)
        return extracted_data

# ============================================================================= #
# ============================================================================= #
# ============================================================================= #
# ============================================================================= #

    def embed_encoded_data_into_DCT(encoded_bits, dct_blocks):
        data_complete = False; encoded_bits.pos = 0
        encoded_data_len = bitstring.pack('uint:32', len(encoded_bits))
        converted_blocks = []
        for current_dct_block in dct_blocks:
            for i in range(1, len(current_dct_block)):
                curr_coeff = np.int32(current_dct_block[i])
                if (curr_coeff > 1):
                    curr_coeff = np.uint8(current_dct_block[i])
                    if (encoded_bits.pos == (len(encoded_bits) - 1)): data_complete = True; break
                    pack_coeff = bitstring.pack('uint:8', curr_coeff)
                    if (encoded_data_len.pos <= len(encoded_data_len) - 1): pack_coeff[-1] = encoded_data_len.read(1)
                    else: pack_coeff[-1] = encoded_bits.read(1)
                    # Replace converted coefficient
                    current_dct_block[i] = np.float32(pack_coeff.read('uint:8'))
            converted_blocks.append(current_dct_block)
        
        if not(data_complete): raise ValueError("Data didn't fully embed into cover image!")
        return converted_blocks


HORIZ_AXIS = 1
VERT_AXIS  = 0

# Standard quantization table as defined by JPEG
JPEG_STD_LUM_QUANT_TABLE = np.asarray([
                                        [16, 11, 10, 16,  24, 40,   51,  61],
                                        [12, 12, 14, 19,  26, 58,   60,  55],
                                        [14, 13, 16, 24,  40, 57,   69,  56],
                                        [14, 17, 22, 29,  51, 87,   80,  62],
                                        [18, 22, 37, 56,  68, 109, 103,  77],
                                        [24, 36, 55, 64,  81, 104, 113,  92],
                                        [49, 64, 78, 87, 103, 121, 120, 101],
                                        [72, 92, 95, 98, 112, 100, 103,  99]
                                      ],
                                      dtype = np.float32)
# Image container class
class YCC_Image(object):
    def __init__(self, cover_image):
        self.height, self.width = cover_image.shape[:2]
        self.channels = [
                         split_image_into_8x8_blocks(cover_image[:,:,0]),
                         split_image_into_8x8_blocks(cover_image[:,:,1]),
                         split_image_into_8x8_blocks(cover_image[:,:,2]),
                        ]

#====================================================================================================#
#====================================================================================================#

def stitch_8x8_blocks_back_together(Nc, block_segments):
    '''
    Take the array of 8x8 pixel blocks and put them together by row so the numpy.block() method can sitch it back together
    :param Nc: Number of pixels in the image (length-wise)
    :param block_segments:
    :return:
    '''
    image_rows = []
    temp = []
    for i in range(len(block_segments)):
        if i > 0 and not(i % int(Nc / 8)):
            image_rows.append(temp)
            temp = [block_segments[i]]
        else:
            temp.append(block_segments[i])
    image_rows.append(temp)

    return np.block(image_rows)

#====================================================================================================#
#====================================================================================================#

def split_image_into_8x8_blocks(image):
    blocks = []
    for vert_slice in np.vsplit(image, int(image.shape[0] / 8)):
        for horiz_slice in np.hsplit(vert_slice, int(image.shape[1] / 8)):
            blocks.append(horiz_slice)
    return blocks


def zigzag(input):
	#initializing the variables
	#----------------------------------
	h = 0
	v = 0

	vmin = 0
	hmin = 0

	vmax = input.shape[0]
	hmax = input.shape[1]

	#print(vmax ,hmax )

	i = 0

	output = np.zeros(( vmax * hmax))
	#----------------------------------

	while ((v < vmax) and (h < hmax)):

		if ((h + v) % 2) == 0:                 # going up

			if (v == vmin):
				#print(1)
				output[i] = input[v, h]        # if we got to the first line

				if (h == hmax):
					v = v + 1
				else:
					h = h + 1

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[i] = input[v, h]
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[i] = input[v, h]
				v = v - 1
				h = h + 1
				i = i + 1


		else:                                    # going down

			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[i] = input[v, h]
				h = h + 1
				i = i + 1

			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[i] = input[v, h]

				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1

				i = i + 1

			elif ((v < vmax -1) and (h > hmin)):     # all other cases
				#print(6)
				output[i] = input[v, h]
				v = v + 1
				h = h - 1
				i = i + 1




		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)
			output[i] = input[v, h]
			break

	#print ('v:',v,', h:',h,', i:',i)
	return output




# Inverse zigzag scan of a matrix
# Arguments are: a 1-by-m*n array,
# where m & n are vertical & horizontal sizes of an output matrix.
# Function returns a two-dimensional matrix of defined sizes,
# consisting of input array items gathered by a zigzag method.
#



def inverse_zigzag(input, vmax, hmax):

	#print input.shape

	# initializing the variables
	#----------------------------------
	h = 0
	v = 0

	vmin = 0
	hmin = 0

	output = np.zeros((vmax, hmax))

	i = 0
	#----------------------------------

	while ((v < vmax) and (h < hmax)):
		#print ('v:',v,', h:',h,', i:',i)
		if ((h + v) % 2) == 0:                 # going up

			if (v == vmin):
				#print(1)

				output[v, h] = input[i]        # if we got to the first line

				if (h == hmax):
					v = v + 1
				else:
					h = h + 1

				i = i + 1

			elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
				#print(2)
				output[v, h] = input[i]
				v = v + 1
				i = i + 1

			elif ((v > vmin) and (h < hmax -1 )):    # all other cases
				#print(3)
				output[v, h] = input[i]
				v = v - 1
				h = h + 1
				i = i + 1


		else:                                    # going down

			if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
				#print(4)
				output[v, h] = input[i]
				h = h + 1
				i = i + 1

			elif (h == hmin):                  # if we got to the first column
				#print(5)
				output[v, h] = input[i]
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1

			elif((v < vmax -1) and (h > hmin)):     # all other cases
				output[v, h] = input[i]
				v = v + 1
				h = h - 1
				i = i + 1




		if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
			#print(7)
			output[v, h] = input[i]
			break


	return output


def DCT_encode(self,image_name,data,filename):
    raw_cover_image = cv2.imread(image_name, flags=cv2.IMREAD_COLOR)
    height, width   = raw_cover_image.shape[:2]
    # Force Image Dimensions to be 8x8 compliant
    while(height % 8): height += 1 # Rows
    while(width  % 8): width  += 1 # Cols
    valid_dim = (width, height)
    padded_image    = cv2.resize(raw_cover_image, valid_dim)
    cover_image_f32 = np.float32(padded_image)
    cover_image_YCC = YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

    # Placeholder for holding stego image data
    stego_image = np.empty_like(cover_image_f32)
    
    for chan_index in range(3):
        # FORWARD DCT STAGE
        dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

        # QUANTIZATION STAGE
        dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

        # Sort DCT coefficients by frequency
        sorted_coefficients = [zigzag(block) for block in dct_quants]

        # Embed data in Luminance layer
        if (chan_index == 0):
            # DATA INSERTION STAGE
            secret_data = ""
            for char in data.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
            embedded_dct_blocks   = embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
            desorted_coefficients = [inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
        else:
            # Reorder coefficients to how they originally were
            desorted_coefficients = [inverse_zigzag(block, vmax=8,hmax=8) for block in sorted_coefficients]

        # DEQUANTIZATION STAGE
        dct_dequants = [np.multiply(data, JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]

        # Inverse DCT Stage
        idct_blocks = [cv2.idct(block) for block in dct_dequants]

        # Rebuild full image channel
        stego_image[:,:,chan_index] = np.asarray(stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

        # Convert back to RGB (BGR) Colorspace
    stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)

    # Clamp Pixel Values to [0 - 255]
    final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))

    # Write stego image
    cv2.imwrite(image_name, final_stego_image)

def extract_text(self,image_name):
    stego_image     = cv2.imread(image_name, flags=cv2.IMREAD_COLOR)
    stego_image_f32 = np.float32(stego_image)
    stego_image_YCC = YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

    # FORWARD DCT STAGE
    dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]  # Only care about Luminance layer

    # QUANTIZATION STAGE
    dct_quants = [np.around(np.divide(item, JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

    # Sort DCT coefficients by frequency
    sorted_coefficients = [zigzag(block) for block in dct_quants]

    # DATA EXTRACTION STAGE
    recovered_data = extract_encoded_data_from_DCT(sorted_coefficients)

    # Determine length of secret message
    data_len = int(recovered_data.read('uint:32') / 8)

    # Extract secret message from DCT coefficients
    extracted_data = bytes()
    for _ in range(data_len): extracted_data += struct.pack('>B', recovered_data.read('uint:8'))

    # Print secret message back to the user
    print(extracted_data.decode('ascii'))