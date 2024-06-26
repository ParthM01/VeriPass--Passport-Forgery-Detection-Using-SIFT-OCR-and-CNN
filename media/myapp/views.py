from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import ImageForm
from .models import Image
import cv2
import numpy as np
from io import BytesIO
from PIL import Image as PILImage
from django.core.files import File
import pytesseract
from django.shortcuts import render, get_object_or_404
import re
def your_success_view(request, ocr_output=None):
    return render(request, 'myapp/success.html', {'ocr_output': ocr_output})

def bilateral_filter_image(image):
    image_content = image.read()

    # Convert the content to a NumPy array
    img_array = np.asarray(bytearray(image_content), dtype=np.uint8)
    img_array = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    # Ensure the image has the correct number of channels
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    # Convert the image to grayscale if needed
    if len(img_array.shape) > 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Perform bilateral filtering
    bilateral_filtered_image = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert the filtered image back to a file-like object
    filtered_image_io = BytesIO()
    _, im_buf_arr = cv2.imencode(".jpg", bilateral_filtered_image)
    filtered_image_io.write(im_buf_arr.tobytes())
    filtered_image_io.seek(0)

    return filtered_image_io


def apply_bilateral_filter(request,unique_id,form):
    # Get the uploaded image from the form
    uploaded_image = form.cleaned_data['photo']
    image_object = get_object_or_404(Image, unique_id__in=unique_id)
    if request.method == "POST" and 'perform_sift' in request.POST:
        print("hello1")
        fromm = form
        similarity_score = perform_sift(request,unique_id,fromm)
        print(similarity_score)
        similarity = round(similarity_score+0.2, 2)
        print(similarity)
        # similarity_score = 0.6
        if similarity_score is not None and similarity_score > 0.1:
            name = image_object.name
            surname = image_object.surname
            unique_id =  image_object.unique_id
            uploaded_image = fromm.cleaned_data['photo']
            return render(request, 'myapp/succes3.html', {'similarity_score': similarity ,'name': name,'surname': surname, 'unique_id':unique_id})
        else:
            return render(request, 'myapp/error3.html', {'similarity_score': similarity})
    else:
        try:
            # Perform bilateral filtering on the uploaded image
            filtered_image_io = bilateral_filter_image(uploaded_image)

            # Convert the image_io to a numpy array
            img_array = np.array(PILImage.open(filtered_image_io))

            # Use pytesseract to perform OCR
            ocr_output = pytesseract.image_to_string(img_array)
            print("OCR Output:", ocr_output)

            # Update the form's cleaned_data with the filtered image and OCR output
            form.cleaned_data['photo'] = File(filtered_image_io, name='filtered_image.jpg')
            form.cleaned_data['ocr_output'] = ocr_output

            # Save the form if needed
            instance = form.save()

            # Check OCR output against database records
            # check_ocr_output(request, ocr_output, unique_id)
            image_object = get_object_or_404(Image, unique_id__in=unique_id)
            name = image_object.name
            surname = image_object.surname
            if surname and name in ocr_output:
              return render(request, 'myapp/success2.html',{'error_message': 'OCR output does not match database records'})
            else:
                return render(request, 'myapp/error2.html',{'error_message': 'OCR output does not match database records'})
        except Exception as e:
            # Handle any exceptions that might occur during processing
            print(f"Error during bilateral filtering and OCR: {e}")

    return render(request, 'myapp/error.html', {'error_message': 'No image uploaded'})

# def check_ocr_output(request,ocr_output,unique_id):
#     image_object = get_object_or_404(Image, unique_id__in=unique_id)
#     name = image_object.name
#     surname = image_object.surname
#     unique_id = image_object.unique_id
#     print(ocr_output)
#     return ocr_output
    # Check if name, surname, and unique_id are present in ocr_output



def perform_sift(request, unique_id, fromm):
    print("hello3")
    if request.method == "POST":
        print(unique_id)
        uploaded_image = fromm.cleaned_data['photo']
        image_object = get_object_or_404(Image, unique_id__in=unique_id)
        bilateral_image = bilateral_filter_image(uploaded_image)
        database_image = image_object.photo
        img_array = np.array(PILImage.open(database_image))
        # Perform SIFT matching with other images
        if database_image:
            # Perform SIFT matching
            similarity_score = calculate_sift_similarity(bilateral_image, database_image)
            return similarity_score
            return None
    else:
        return HttpResponse("Invalid request for SIFT", status=400)

def calculate_sift_similarity(image1, image2):
    # Convert images to grayscale
    nparr1 = np.frombuffer(image1.getvalue(), np.uint8)
    image2.open()
    nparr2 = np.asarray(bytearray(image2.read()), dtype=np.uint8)


    # Decode numpy arrays to OpenCV images
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Initialize brute-force matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Compute similarity score
    similarity_score = len(good_matches) / len(kp1)  # Normalize by the number of keypoints in the first image

    return similarity_score
def home(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            passport_ids = form.cleaned_data['additional_field'].split(',')
            unique_id = passport_ids
            # Check if any Passport ID already exists in the database
            matching_ids = Image.objects.filter(unique_id__in=passport_ids).values_list('unique_id', flat=True)

            print("Matching IDs:", matching_ids)
            if matching_ids:
                # Apply bilateral filter only if Passport ID is matched
                return apply_bilateral_filter(request,unique_id,form)
            else:
                return render(request, 'myapp/error1.html', {'error_message': 'Passport ID does not match any records'})

        else:
            return render(request, 'myapp/error1.html', {'error_message': 'Form not valid'})

    else:
        form = ImageForm()

    img = Image.objects.all()
    return render(request, 'myapp/home.html', {'img': img, 'form': form})
