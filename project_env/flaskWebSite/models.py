from flaskWebSite import db

class INOUT(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    InputtedPic = db.Column(db.String(100), nullable=True)
    OutputtedPicOne = db.Column(db.String(100), nullable=True)
    OutputtedPicTwo = db.Column(db.String(100), nullable=True)
    SelectedPic = db.Column(db.String(100), nullable=True)
    text = db.Column(db.Text, unique=False, nullable=True)
    
    def __repr__(self):
        return '<id %r>' % self.id
    


class ClassificationImgs(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    InputtedPic = db.Column(db.String(100), nullable=True)
    prediction = db.Column(db.String(100), unique=False, nullable=True)
    feedback = db.Column(db.String(100), unique=False, nullable=True)
    
    def __repr__(self):
        return '<id %r>' % self.id