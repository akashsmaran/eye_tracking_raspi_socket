import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np
import time
import imutils

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 6322))
server_socket.listen(0)
prev = 0
# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
try:
	while True:
	# Read the length of the image as a 32-bit unsigned int. If the
	# length is zero, quit the loop
		image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
		if not image_len:
			break
		# Construct a stream to hold the image data and read the image
		# data from the connection
		image_stream = io.BytesIO()
		image_stream.write(connection.read(image_len))
		# Rewind the stream, open it as an image with PIL and do some
		# processing on it
		image_stream.seek(0)
		image = Image.open(image_stream)
		cv_image = np.array(image)
		cv2.imshow('Streamright',cv_image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if time.time()-prev>=3:
			prev = time.time()
			gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 0)
			y = gray.copy()
	
			###		
			#cv2.imshow("gray_image", gray)
			#cv2.waitKey(0)

			ret,th1 = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
			kernel = np.ones((25,25), 'uint8')
			temp_mask = cv2.dilate(th1, kernel)
			gray = cv2.inpaint(gray, temp_mask, 1, cv2.INPAINT_TELEA)
		
			#cv2.imshow("inpainted_image", gray)
			#cv2.waitKey(0)
		
		
			###
			#th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
		    	#	cv2.THRESH_BINARY,11,2)
			###
			#ret, thresh1 = cv2.threshold(y,60,70,cv2.THRESH_BINARY)
			thresh1 = cv2.inRange(gray,60,70)
			cv2.imshow("thresholded_image", thresh1)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
			#cv2.waitKey(0)
			###
		
			#lower_black = (120,120,120)   
			#upper_black = (245,245,245)
			#mask = cv2.inRange(gray, lower_black,upper_black)

			#cv2.imshow("image", mask)
			#cv2.waitKey(0)

			#edged = cv2.Canny(gray, 10, 20 )
			edged = cv2.dilate(thresh1, None, iterations = 5)
			edged = cv2.erode(edged, None, iterations=5)
			edged = cv2.Canny(edged, 10 ,40)
		
			#cv2.imshow("image", edged)
			#cv2.waitKey(0)


			cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if imutils.is_cv2() else cnts[1]
			try:
				(cnts, _) = contours.sort_contours(cnts)
				pixelsPerMetric = None

	
		    # loop over the contours individually
				for c in cnts:
					# if the contour is not sufficiently large, ignore it
					if cv2.contourArea(c) < 100:
						continue
					orig = cv_image.copy()
					box = cv2.minAreaRect(c)
					box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
					box = np.array(box, dtype="int")

					box = perspective.order_points(box)
					cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

		
					for (x, y) in box:
						cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

					(tl, tr, br, bl) = box
					(tltrX, tltrY) = midpoint(tl, tr)
					(blbrX, blbrY) = midpoint(bl, br)
					(tlblX, tlblY) = midpoint(tl, bl)
					(trbrX, trbrY) = midpoint(tr, br)

					cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
					cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
					cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
					cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

					# draw lines between the midpoints
					cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
					    (255, 0, 255), 2)
					cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
					    (255, 0, 255), 2)

					# compute the Euclidean distance between the midpoints
					dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
					dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
					if pixelsPerMetric is None:
						pixelsPerMetric = dB / 1.2

					# compute the size of the object
					dimA = dA / pixelsPerMetric
					dimB = dB / pixelsPerMetric
					r.append(min(dimA/2,dimB/2))

					# draw the object sizes on the image
					cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
					cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
					#cv2.imwrite("orig", orig)
					cv2.namedWindow("image", cv2.WINDOW_NORMAL)
					#cv2.imshow("image", orig)
					#cv2.waitKey(0)
				
					#IMAGE PROCESSING ENDS
				#cv2.imwrite("/home/akash/Desktop/response.png", orig)

			except:
				#cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
				circles = cv2.HoughCircles(y,cv2.HOUGH_GRADIENT,1,220,
						    param1=50,param2=20,minRadius=20,maxRadius=50)
				lr = []

				if circles is None:
					print "No circles found"
				else:
				   	circles = np.uint16(np.around(circles))

				   	for i in circles[0,:]:
				      # draw the outer circle
				      		cv2.circle(x,(i[0],i[1]),i[2],(0,255,0),2)
				      # draw the center of the circle
				      		cv2.circle(x,(i[0],i[1]),2,(0,0,255),3)
				      		lr.append(i[2])
				   	rx = min(lr)/240.0
					r.append(rx)

				   	#cv2.imshow('detected circles',x)
					#cv2.waitKey(0)


finally:
	print "abc"
	connection.close()
	server_socket.close()
