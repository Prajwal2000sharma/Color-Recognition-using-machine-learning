clicked=False
r=g=b=xpos=ypos=0
source_image=[]

def major():
    x=int(input(("Enter Choice:\n1-Image data Set\n2-Predefined Dataset\n0-exit\n")))
    while(1):
        if x==1:
            major1()
        elif x==2:
            major2()
        elif x==0:
            exit(0)
        else:
            print("Wrong Choice")
        x=int(input(("Enter Choice:\n1-Image data Set\n2-Predefined Dataset\n0-exit\n")))

def major1():
    import csv
    import random
    import math
    import operator
    import cv2
    #from PIL import image
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import itemfreq
    import os.path
    import sys

    # calculation of euclidead distance
    def calculateEuclideanDistance(variable1,variable2,length):
        distance=0
        for x in range(length):
            distance=distance+pow(variable1[x]-variable2[x],2)
        return math.sqrt(distance)

    # get k nearest neigbors
    def kNearestNeighbors(training_feature_vector,testInstance,k):
        distances=[]
        length=len(testInstance)
        for x in range(len(training_feature_vector)):
            dist=calculateEuclideanDistance(testInstance,
                    training_feature_vector[x],length)
            distances.append((training_feature_vector[x],dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors=[]
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    # votes of neighbors
    def responseOfNeighbors(neighbors):
        all_possible_neighbors={}
        for x in range(len(neighbors)):
            response=neighbors[x][-1]
            if response in all_possible_neighbors:
                all_possible_neighbors[response]=all_possible_neighbors[response]+1
            else:
                all_possible_neighbors[response]=1
        sortedVotes=sorted(all_possible_neighbors.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]

    # Load image feature data to training feature vectors and test feature vector
    def loadDataset(filename,filename2,training_feature_vector=[],test_feature_vector=[],):
        with open(filename) as csvfile:
            lines=csv.reader(csvfile)
            dataset=list(lines)
            for x in range(len(dataset)):
                for y in range(3):
                    dataset[x][y]=float(dataset[x][y])
                training_feature_vector.append(dataset[x])
        with open(filename2) as csvfile:
            lines=csv.reader(csvfile)
            dataset=list(lines)
            for x in range(len(dataset)):
                for y in range(3):
                    dataset[x][y]=float(dataset[x][y])
                test_feature_vector.append(dataset[x])
                
    def main(training_data,test_data):
        training_feature_vector=[]  
        test_feature_vector=[]  
        loadDataset(training_data,test_data,training_feature_vector,test_feature_vector)
        classifier_prediction=[]  
        k=3  
        for x in range(len(test_feature_vector)):
            neighbors=kNearestNeighbors(training_feature_vector,test_feature_vector[x],k)
            result=responseOfNeighbors(neighbors)
            classifier_prediction.append(result)
        return classifier_prediction[0]

    def color_histogram_of_test_image(test_src_image):
        # load the image
        image=test_src_image
        chans=cv2.split(image)
        colors=('b','g','r')
        features=[]
        feature_data=''
        counter=0
        for (chan,color) in zip(chans,colors):
            counter=counter+1
            hist=cv2.calcHist([chan],[0],None,[256],[0,256])
            features.extend(hist)
            # find the peak pixel values for R,G,and B
            elem=np.argmax(hist)
            if counter == 1:
                blue=str(elem)
            elif counter == 2:
                green=str(elem)
            elif counter == 3:
                red=str(elem)
                feature_data=red+','+green+','+blue
                # print(feature_data)
        with open('test.data','w') as myfile:
            myfile.write(feature_data)
            
    def color_histogram_of_training_image(img_name):
        # detect image color by using image file name to label training data
        if 'red' in img_name:
            data_source='red'
        elif 'yellow' in img_name:
            data_source='yellow'
        elif 'green' in img_name:
            data_source='green'
        elif 'orange' in img_name:
            data_source='orange'
        elif 'white' in img_name:
            data_source='white'
        elif 'black' in img_name:
            data_source='black'
        elif 'blue' in img_name:
            data_source='blue'
        elif 'violet' in img_name:
            data_source='violet'
        elif 'gray' in img_name:
            data_source='gray'
        elif 'pink' in img_name:
            data_source='pink'
        elif 'sky' in img_name:
            data_source='sky'
        elif 'brown' in img_name:
            data_source='brown'
        # load the image
        image=cv2.imread(img_name)
        chans=cv2.split(image)
        colors=('b','g','r')
        features=[]
        feature_data=''
        counter=0
        for (chan,color) in zip(chans,colors):
            counter=counter+1
            hist=cv2.calcHist([chan],[0],None,[256],[0,256])
            features.extend(hist)
            # find the peak pixel values for R,G,and B
            elem=np.argmax(hist)
            if counter==1:
                blue=str(elem)
            elif counter==2:
                green=str(elem)
            elif counter==3:
                red=str(elem)
                feature_data=red+','+green+','+blue
        with open('training.data','a') as myfile:
            myfile.write(feature_data+','+data_source+'\n')

    def training():
        # red color training images
        for f in os.listdir('./training_dataset/red'):
            color_histogram_of_training_image('./training_dataset/red/'+f)
        # yellow color training images
        for f in os.listdir('./training_dataset/yellow'):
            color_histogram_of_training_image('./training_dataset/yellow/'+f)
        # green color training images
        for f in os.listdir('./training_dataset/green'):
            color_histogram_of_training_image('./training_dataset/green/'+f)
        # orange color training images
        for f in os.listdir('./training_dataset/orange'):
            color_histogram_of_training_image('./training_dataset/orange/'+f)
        # white color training images
        for f in os.listdir('./training_dataset/white'):
            color_histogram_of_training_image('./training_dataset/white/'+f)
        # black color training images
        for f in os.listdir('./training_dataset/black'):
            color_histogram_of_training_image('./training_dataset/black/'+f)
        # violet color training images
        for f in os.listdir('./training_dataset/violet'):
            color_histogram_of_training_image('./training_dataset/violet/'+f)
        # blue color training images
        for f in os.listdir('./training_dataset/blue'):
            color_histogram_of_training_image('./training_dataset/blue/'+f)
        # sky blue color training images
        for f in os.listdir('./training_dataset/sky'):
            color_histogram_of_training_image('./training_dataset/sky/'+f)
        # pink color training images
        for f in os.listdir('./training_dataset/pink'):
            color_histogram_of_training_image('./training_dataset/pink/'+f)
        # brown color training images
        for f in os.listdir('./training_dataset/brown'):
            color_histogram_of_training_image('./training_dataset/brown/'+f)
        # gray color training images
        for f in os.listdir('./training_dataset/gray'):
            color_histogram_of_training_image('./training_dataset/gray/'+f)   

    def func1():
        source_image=input("Enter image path:")
         # read the test image
        try:
            source_image=cv2.imread(sys.argv[1])
        except:
            source_image=cv2.imread(source_image)
        prediction='n.a.'
        # checking whether the training data is ready
        PATH='./training.data'
        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('training data is ready,classifier is loading...')
        else:
            print ('training data is being created...')
            open('training.data','w')
            training()
            print ('training data is ready,classifier is loading...')
        # get the prediction
        color_histogram_of_test_image(source_image)
        prediction=main('training.data','test.data')
        print('Detected color is:',prediction)
        cv2.putText(source_image,'Prediction: '+prediction,(15,45),cv2.FONT_HERSHEY_PLAIN,3,200)
        # Display the resulting frame
        cv2.imshow('major colour in image',source_image)
        cv2.waitKey(0)

    clicked=False
    r=g=b=xpos=ypos=0
    source_image=[]
    def draw_function(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            global b,g,r,xpos,ypos,clicked,source_image
            clicked=True
            xpos=x
            ypos=y
            b,g,r=source_image[y,x]
            b=int(b)
            g=int(g)
            r=int(r)
    def func2():
        global b,g,r,xpos,ypos,clicked,source_image
        source_image=input("Enter image path:")
        # read the test image
        try:
            source_image=cv2.imread(sys.argv[1])
        except:
            source_image=cv2.imread(source_image)
        prediction='n.a.'
        # checking whether the training data is ready
        PATH='./training.data'    
        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('training data is ready,classifier is loading...')
        else:
            print ('training data is being created...')
            open('training.data','w')
            training()
            print ('training data is ready,classifier is loading...')
        cv2.namedWindow('loaded image')
        cv2.setMouseCallback('loaded image',draw_function)
        while(1):
            cv2.imshow("loaded image",source_image)
            if (clicked):
                #cv2.rectangle(image,startpoint,endpoint,color,thickness) -1 thickness fills rectangle entirely
                cv2.rectangle(source_image,(20,20),(750,60),(211,211,211),-1)
                #Creating text string to display ( Color name and RGB values )
                str_=str(r)+','+str(g)+','+str(b)
                with open('test.data','w') as myfile:
                    myfile.write(str_)
                prediction =main('training.data','test.data')
                text=prediction
                #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType,(optional bottomLeft bool))
                cv2.putText(source_image,text,(50,50),2,0.8,(b,g,r),2,cv2.LINE_AA)
                #For very light colours we will display text in black colour
                if(r+g+b>=600):
                    cv2.putText(source_image,text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
                clicked=False
            #Break the loop when user hits 'esc' key 
            if cv2.waitKey(20) & 0xFF ==27:
                break
        cv2.destroyAllWindows()
        
    def func3():
        cnt=0
        global b,g,r,xpos,ypos,clicked,source_image,frame
        cap=cv2.VideoCapture(0)
        (ret,source_image)=cap.read()
        prediction='n.a.'
        # checking whether the training data is ready
        PATH='./training.data'
        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('training data is ready,classifier is loading...')
        else:
            print ('training data is being created...')
            open('training.data','w')
            training()
            print ('training data is ready,classifier is loading...')
        # Capture frame-by-frame
        (ret,source_image)=cap.read()
        cv2.namedWindow('captured image')
        cv2.setMouseCallback('captured image',draw_function)
        while True:
            cnt=cnt+1
            # Capture frame-by-frame
            (ret,source_image)=cap.read()
            while(1):
                cv2.imshow("captured image",source_image)
                if (clicked):
                    #cv2.rectangle(image,startpoint,endpoint,color,thickness) -1 thickness fills rectangle entirely
                    cv2.rectangle(source_image,(20,20),(750,60),(211,211,211),-1)
                    #Creating text string to display ( Color name and RGB values )
                    str_=str(r)+','+str(g)+','+str(b)
                    with open('test.data','w') as myfile:
                        myfile.write(str_)
                    prediction =main('training.data','test.data')
                    text=prediction
                    #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType,(optional bottomLeft bool) )
                    cv2.putText(source_image,text,(50,50),2,0.8,(b,g,r),2,cv2.LINE_AA)
                    #For very light colours we will display text in black colour
                    if(r+g+b>=600):
                        cv2.putText(source_image,text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
                    clicked=False
                #Break the loop when user hits 'esc' key 
                if cv2.waitKey(20) & 0xFF ==27:
                    break
            if cnt>4:
                break
            # When everything done,release the capture
        cap.release()
        cv2.destroyAllWindows()

    def func4():
        cap=cv2.VideoCapture(0)
        (ret,frame)=cap.read()
        prediction='n.a.'
        # checking whether the training data is ready
        PATH='./training.data'
        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('training data is ready,classifier is loading...')
        else:
            print ('training data is being created...')
            open('training.data','w')
            training()
            print ('training data is ready,classifier is loading...')
        while True:
            # Capture frame-by-frame
            (ret,frame)=cap.read()
            l=['white','gray','pink']
            if prediction in l:
                cv2.putText(frame,'Prediction: '+prediction,(15,45),2,0.8,(0,0,0),2,cv2.LINE_AA)
            else:
                cv2.putText(frame,'Prediction: '+prediction,(15,45),2,0.8,(255,255,255),2,cv2.LINE_AA)           
            # Display the resulting frame
            cv2.imshow('webcam colour',frame)
            color_histogram_of_test_image(frame)
            prediction=main('training.data','test.data')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done,release the capture
        cap.release()
        cv2.destroyAllWindows()
        
    x=int(input(("Enter Choice:\n1-Detect Uploaded Image Colour\n2-Detect Colour of selected point on Image\n3-Detect Colour using Webcam capture \n4-Detect Colour using Webcam\n9-Previous Menu\n")))
    while(1):
        if x==1:
            func1()
        elif x==2:
            func2()
        elif x==3:
            func3()
        elif x==4:
            func4()
        elif x==9:
           major()
        else:
            print("Wrong Choice")
        x=int(input(("Enter Choice:\n1-Detect Uploaded Image Colour\n2-Detect Colour of selected point on Image\n3-Detect Colour using Webcam capture \n4-Detect Colour using Webcam\n9-Previous Menu\n")))

def major2():
    import csv
    import random
    import math
    import operator
    import cv2
    from PIL import Image
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import itemfreq
    import os.path
    import sys
    
    # calculation of euclidead distance
    def calculateEuclideanDistance(variable1,variable2,length):
        distance=0
        for x in range(length):
            distance=distance+pow(variable1[x]-variable2[x],2)
        return math.sqrt(distance)

    # get k nearest neigbors
    def kNearestNeighbors(training_feature_vector,testInstance,k):
        distances=[]
        length=len(testInstance)
        for x in range(len(training_feature_vector)):
            dist=calculateEuclideanDistance(testInstance,
                    training_feature_vector[x],length)
            distances.append((training_feature_vector[x],dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors=[]
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    # votes of neighbors
    def responseOfNeighbors(neighbors):
        all_possible_neighbors={}
        for x in range(len(neighbors)):
            response=neighbors[x][-1]
            if response in all_possible_neighbors:
                all_possible_neighbors[response]=all_possible_neighbors[response]+1
            else:
                all_possible_neighbors[response]=1
        sortedVotes=sorted(all_possible_neighbors.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]

    # Load image feature data to training feature vectors and test feature vector
    def loadDataset(filename,filename2,training_feature_vector=[],test_feature_vector=[],):
        with open(filename) as csvfile:
            lines=csv.reader(csvfile)
            dataset=list(lines)
            for x in range(len(dataset)):
                for y in range(3):
                    dataset[x][y]=float(dataset[x][y])
                training_feature_vector.append(dataset[x])
        with open(filename2) as csvfile:
            lines=csv.reader(csvfile)
            dataset=list(lines)
            for x in range(len(dataset)):
                for y in range(3):
                    dataset[x][y]=float(dataset[x][y])
                test_feature_vector.append(dataset[x])
                
    def main(training_data,test_data):
        training_feature_vector=[]  
        test_feature_vector=[]  
        loadDataset(training_data,test_data,training_feature_vector,test_feature_vector)
        classifier_prediction=[]  
        k=3  
        for x in range(len(test_feature_vector)):
            neighbors=kNearestNeighbors(training_feature_vector,test_feature_vector[x],k)
            result=responseOfNeighbors(neighbors)
            classifier_prediction.append(result)
        return classifier_prediction[0]

    def color_histogram_of_test_image(test_src_image):
        # load the image
        image=test_src_image
        chans=cv2.split(image)
        colors=('b','g','r')
        features=[]
        feature_data=''
        counter=0
        for (chan,color) in zip(chans,colors):
            counter=counter+1
            hist=cv2.calcHist([chan],[0],None,[256],[0,256])
            features.extend(hist)
            # find the peak pixel values for R,G,and B
            elem=np.argmax(hist)
            if counter == 1:
                blue=str(elem)
            elif counter == 2:
                green=str(elem)
            elif counter == 3:
                red=str(elem)
                feature_data=red+','+green+','+blue
                # print(feature_data)
        with open('test2.data','w') as myfile:
            myfile.write(feature_data)
            
            
    def func1():
        source_image=input("Enter image path:")
         # read the test image
        try:
            source_image=cv2.imread(sys.argv[1])
        except:
            source_image=cv2.imread(source_image)
        prediction='n.a.'
        # checking whether the training data is ready
        PATH='./colordata.data'
        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('Trained data found and ready')
        else:
            print("File Not Found")
            exit(0)
        # get the prediction
        color_histogram_of_test_image(source_image)
        prediction=main('colordata.data','test2.data')
        print('Detected color is:',prediction)
        cv2.putText(source_image,'Prediction: '+prediction,(15,45),cv2.FONT_HERSHEY_PLAIN,1,1)
        # Display the resulting frame
        cv2.imshow('color classifier',source_image)
        cv2.waitKey(0)

    clicked=False
    r=g=b=xpos=ypos=0
    source_image=[]
    def draw_function(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            global b,g,r,xpos,ypos,clicked,source_image
            clicked=True
            xpos=x
            ypos=y
            b,g,r=source_image[y,x]
            b=int(b)
            g=int(g)
            r=int(r)
    def func2():
        global b,g,r,xpos,ypos,clicked,source_image
        source_image=input("Enter image path:")
        # read the test image
        try:
            source_image=cv2.imread(sys.argv[1])
        except:
            source_image=cv2.imread(source_image)
        prediction='n.a.'
        # checking whether the training data is ready
        PATH='./colordata.data'    
        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('Trained data found and ready')
        else:
            print("File Not Found")
            exit(0)
        cv2.namedWindow('loaded image')
        cv2.setMouseCallback('loaded image',draw_function)
        while(1):
            cv2.imshow("loaded image",source_image)
            if (clicked):
                #cv2.rectangle(image,startpoint,endpoint,color,thickness) -1 thickness fills rectangle entirely
                cv2.rectangle(source_image,(20,20),(750,60),(211,211,211),-1)
                #Creating text string to display ( Color name and RGB values )
                str_=str(r)+','+str(g)+','+str(b)
                with open('test2.data','w') as myfile:
                    myfile.write(str_)
                prediction =main('colordata.data','test2.data')
                text=prediction
                #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType,(optional bottomLeft bool))
                cv2.putText(source_image,text,(50,50),2,0.8,(b,g,r),2,cv2.LINE_AA)
                #For very light colours we will display text in black colour
                if(r+g+b>=600):
                    cv2.putText(source_image,text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
                clicked=False
            #Break the loop when user hits 'esc' key 
            if cv2.waitKey(20) & 0xFF ==27:
                break
        cv2.destroyAllWindows()
    def func3():
        cnt=0
        global b,g,r,xpos,ypos,clicked,source_image,frame
        cap=cv2.VideoCapture(0)
        (ret,source_image)=cap.read()
        prediction='n.a.'
        # checking whether the training data is ready
        PATH='./colordata.data'
        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('Trained data found and ready')
        else:
            print("File Not Found")
            exit(0)
        # Capture frame-by-frame
        (ret,source_image)=cap.read()
    ##        cv2.putText(frame,'Prediction: '+prediction,(15,45),cv2.FONT_HERSHEY_PLAIN,3,200,)
        cv2.namedWindow('captured image')
        cv2.setMouseCallback('captured image',draw_function)
        while True:
            cnt=cnt+1
            # Capture frame-by-frame
            (ret,source_image)=cap.read()
            while(1):
                cv2.imshow("captured image",source_image)
                if (clicked):
                    #cv2.rectangle(image,startpoint,endpoint,color,thickness) -1 thickness fills rectangle entirely
                    cv2.rectangle(source_image,(20,20),(750,60),(211,211,211),-1)
                    #Creating text string to display ( Color name and RGB values )
                    str_=str(r)+','+str(g)+','+str(b)
                    with open('test2.data','w') as myfile:
                        myfile.write(str_)
                    prediction =main('colordata.data','test2.data')
                    text=prediction
                    #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType,(optional bottomLeft bool) )
                    cv2.putText(source_image,text,(50,50),2,0.8,(b,g,r),2,cv2.LINE_AA)
                    #For very light colours we will display text in black colour
                    if(r+g+b>=600):
                        cv2.putText(source_image,text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
                    clicked=False
                #Break the loop when user hits 'esc' key 
                if cv2.waitKey(20) & 0xFF ==27:
                    break
            if cnt>4:
                break
            # When everything done,release the capture
        cap.release()
        cv2.destroyAllWindows()

    def func4():
        cap=cv2.VideoCapture(0)
        (ret,frame)=cap.read()
        prediction='n.a.'

        # checking whether the training data is ready
        PATH='./colordata.data'

        if os.path.isfile(PATH) and os.access(PATH,os.R_OK):
            print ('Trained data found and ready')
        else:
            print("File Not Found")
            exit(0)

        while True:
            # Capture frame-by-frame
            (ret,frame)=cap.read()
            l=['white','gray','pink']
            if prediction in l:
                cv2.putText(frame,'Prediction: '+prediction,(15,45),2,0.8,(0,0,0),2,cv2.LINE_AA)
            else:
                cv2.putText(frame,'Prediction: '+prediction,(15,45),2,0.8,(255,255,255),2,cv2.LINE_AA)           
            # Display the resulting frame
            cv2.imshow('color classifier',frame)
            color_histogram_of_test_image(frame)
            prediction=main('colordata.data','test2.data')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done,release the capture
        cap.release()
        cv2.destroyAllWindows()
        
    x=int(input(("Enter Choice:\n1-Detect Uploaded Image Colour\n2-Detect Colour of selected point on Image\n3-Detect Colour using Webcam capture \n4-Detect Colour using Webcam\n9-Previous Menu\n")))
    while(1):
        if x==1:
            func1()
        elif x==2:
            func2()
        elif x==3:
            func3()
        elif x==4:
            func4()
        elif x==9:
            major()
        else:
            print("Wrong Choice")
        x=int(input(("Enter Choice:\n1-Detect Uploaded Image Colour\n2-Detect Colour of selected point on Image\n3-Detect Colour using Webcam capture \n4-Detect Colour using Webcam\n9-Previous Menu\n")))

x=int(input(("Enter Choice:\n1-Image data Set\n2-Predefined Dataset\n0-exit\n")))
if x==1:
    major1()
elif x==2:
    major2()
elif x==0:
    exit(0)    
else:
    print("Wrong Choice")
