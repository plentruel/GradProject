from flask import render_template, redirect, url_for, flash
from flaskWebSite.models import INOUT, ClassificationImgs
from flaskWebSite.forms import UploadImgForm, SelectStuffForm, UploadImg, FeedbackForm
from flaskWebSite.utils import save_picture, predict_single_image, classes, generate
from flaskWebSite import app, db


@app.route("/", methods=['GET','POST'])
def home():
    
    form = UploadImgForm()
    

    if form.validate_on_submit():

        return redirect(url_for('home'))
    
    return render_template("index.html", form=form, title = 'Ana Sayfa')

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    form = UploadImg()
    feedback_form = FeedbackForm()  # Initialize the feedback form

    if form.validate_on_submit():
        # Save the uploaded image and get the prediction
        picture_file = save_picture(form.img.data, 'static/classificationimages')
        image_path = "project_env/flaskWebSite/static/classificationimages/" + picture_file
        pred = predict_single_image(image_path)
        
        # Add the image and prediction to the database
        addable = ClassificationImgs(InputtedPic=picture_file, prediction=pred)
        db.session.add(addable)
        db.session.commit()

        # After predicting, show the feedback form
        return render_template("class.html", form=form, feedback_form=feedback_form, title='Classification', pred=pred, classes=classes, img_path=image_path)

    # If the feedback form is submitted
    if feedback_form.validate_on_submit():
        # Retrieve the latest image and prediction entry
        latest_img = ClassificationImgs.query.order_by(ClassificationImgs.id.desc()).first()
        if latest_img:
            # Save the feedback to the database
            latest_img.feedback = feedback_form.feedback.data
            db.session.commit()

        return render_template("class.html", form=form, feedback_form=feedback_form, title='Classification', pred=latest_img.prediction, classes=classes, img_path=latest_img.InputtedPic)

    return render_template("class.html", form=form, feedback_form=feedback_form, title='Classification', classes=classes)



@app.route("/drawing", methods=['GET','POST'])
def drawing():
    
    
    return render_template("drawing.html", title = 'drawing')


@app.route("/rlhf", methods=['GET','POST'])
def rlhf():
    form = UploadImg()
    if form.validate_on_submit():

        picture_file = save_picture(form.img.data, 'static/images')

        image_path = "project_env/flaskWebSite/static/images/" + picture_file

        outimgae =  generate(image_path)

        return render_template("rlhf.html", form=form, title = 'drawing', outimgae = outimgae)
    
    return render_template("rlhf.html", form=form, title = 'drawing')
