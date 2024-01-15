import cv2    
def draw(persons,violate,img,frame,masked_faces,unmasked_faces,BASE_PATH,width):
    masked_face_count = len(masked_faces)
    unmasked_face_count = len(unmasked_faces)
    for (i, (bbox)) in enumerate(persons):
      a,b,c,d = persons[i]
      if i in violate:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0,100,255), 2)
      else:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0,255,0), 2)
      for f in range(masked_face_count):
       a,b,c,d = masked_faces[f]
       if i in violate:
        color = (255, 255, 133)
       else:
        color = (255,255,153)
        cv2.rectangle(img, (a, b), (c,d), color, 2)
      for f in range(unmasked_face_count):
       a,b,c,d = unmasked_faces[f]
       if i in violate:
        color = (255, 0, 0)
       else:
         color= (0,102,204) 
         cv2.rectangle(img, (a, b), (c,d), color, 2)
          # Monitoring Status
      cv2.rectangle(img,(0,0),(width,50),(0,0,0),-1)
      cv2.rectangle(img,(1,1),(width-1,50),(255,255,255),2)
      cons = 12
          # Count 
      person_count = len(persons)
      masked_face_count = len(masked_faces)
      unmasked_face_count = len(unmasked_faces)
      string = "People = " + str(len(persons))
      cv2.putText(img,string,(cons,34),cv2.FONT_HERSHEY_SIMPLEX,1,(255,128,128),2)
      cons += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][0]
      string = "(" +str(masked_face_count)+" masked "+str(unmasked_face_count)+" not masked "+\
             str(person_count-masked_face_count-unmasked_face_count)+" Unknown)"
      cv2.putText(img,string,(cons,34),cv2.FONT_HERSHEY_SIMPLEX,1,(255,204,0),2)
      cv2.imwrite(BASE_PATH+"Results/Frames/"+str(frame)+'.jpg',img)
      if cv2.waitKey(1) == 27:
       break
    

