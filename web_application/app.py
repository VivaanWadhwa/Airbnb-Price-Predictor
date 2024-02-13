from flask import Flask, render_template, session, url_for, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, SelectField
import joblib
import sklearn
import pandas


app = Flask(__name__)
app.config['SECRET_KEY'] = 'XeAftU_78Fa67*lpUav'

def return_prediction(pipe_lr , newPred):   
    prediction = pipe_lr.predict(newPred)[0]    
    return prediction

model = joblib.load('price_predictor.joblib')

final_numerical_feats = ["minimum_nights", 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']
final_categorical_feats = ['neighbourhood_group', 'room_type', "neighbourhood"]
final_drop_feats = ["id", "host_name","latitude", "longitude", "reviews_per_month", "number_of_reviews_ltm", "license"]

neighbourhood_groups = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
room_types = ["Entire Home/Apt", "Private Room", "Shared Room", "Hotel Room"]
neighbourhoods = ['Clinton Hill', "Hell's Kitchen", 'Chelsea', 'Washington Heights',
       'Murray Hill', 'Williamsburg', 'Sunset Park', 'Astoria',
       'Financial District', 'Sunnyside', 'Midtown', 'East Elmhurst',
       'Upper West Side', 'West Village', 'Kensington', 'Bushwick',
       'Red Hook', 'Tribeca', 'Woodside', 'Roosevelt Island',
       'Concourse Village', 'Harlem', 'Flatbush', 'University Heights',
       'Flushing', 'Greenpoint', 'Ridgewood', 'Cypress Hills',
       "Prince's Bay", 'Port Richmond', 'East Village', 'Clason Point',
       'Bedford-Stuyvesant', 'Gravesend', 'Long Island City', 'Gramercy',
       'East Flatbush', 'East Harlem', 'St. George', 'Boerum Hill',
       'South Ozone Park', 'Park Slope', 'Crown Heights', 'Wakefield',
       'Forest Hills', 'Springfield Gardens', 'North Riverdale',
       'Belmont', 'Corona', 'Tremont', 'Elmhurst', 'Queens Village',
       'Prospect-Lefferts Gardens', 'Fort Greene', 'Allerton',
       'Upper East Side', 'Richmond Hill', 'Kips Bay',
       'Morningside Heights', 'Flatlands', 'St. Albans', 'Bay Ridge',
       'SoHo', 'Lower East Side', 'West Brighton', 'Parkchester',
       'Brownsville', 'Ozone Park', 'Jackson Heights', 'Kew Gardens',
       'Prospect Heights', 'Jamaica', 'Arrochar', 'Kew Gardens Hills',
       'Laurelton', 'Morris Park', 'Rosedale', 'Arverne', 'Woodhaven',
       'Norwood', 'Rosebank', 'Tompkinsville', 'East New York',
       'Far Rockaway', 'Edenwald', 'Baychester', 'Canarsie',
       'Williamsbridge', 'Kingsbridge', 'Rego Park', 'Sheepshead Bay',
       'Ditmars Steinway', 'Melrose', 'DUMBO', 'Theater District',
       'Civic Center', 'Chinatown', 'Flatiron District', 'Nolita',
       'Bath Beach', 'East Morrisania', 'Little Italy', 'Fresh Meadows',
       'Midwood', 'Glendale', 'West Farms', 'Carroll Gardens',
       'Mount Hope', 'Gowanus', 'Windsor Terrace', 'Downtown Brooklyn',
       'Maspeth', 'South Slope', "Bull's Head", 'Navy Yard',
       'Stuyvesant Town', 'Borough Park', 'Dyker Heights',
       'Greenwich Village', 'College Point', 'Rockaway Beach',
       'Mott Haven', 'Oakwood', 'Morris Heights', 'Inwood', 'Cobble Hill',
       'Jamaica Estates', 'Brighton Beach', 'Brooklyn Heights',
       'Vinegar Hill', 'Hollis', 'Jamaica Hills', 'Coney Island',
       'Arden Heights', 'New Springville', 'Bensonhurst', 'Howard Beach',
       'Throgs Neck', 'Cambria Heights', 'Battery Park City', 'Bayside',
       'Briarwood', 'Longwood', 'Shore Acres', 'Fordham', 'Whitestone',
       'Unionport', 'Castle Hill', 'Bergen Beach', 'Bayswater',
       'Port Morris', 'Middle Village', 'NoHo', 'Bellerose', 'Mount Eden',
       'Emerson Hill', 'Pelham Bay', 'Randall Manor', 'Mill Basin',
       'Fort Hamilton', 'Eltingville', 'Woodrow', 'Clifton',
       'New Dorp Beach', 'Grant City', 'Edgemere', 'Lighthouse Hill',
       'South Beach', 'Rossville', 'Belle Harbor', 'New Brighton',
       'Van Nest', 'Hunts Point', 'Westchester Square', 'Spuyten Duyvil',
       'Mariners Harbor', 'Midland Beach', 'Concourse', 'Tottenville',
       'Schuylerville', 'Marble Hill', 'Co-op City', 'Stapleton',
       'Morrisania', 'Claremont Village', 'Two Bridges', 'Concord',
       'Holliswood', 'Dongan Hills', 'City Island', 'Eastchester',
       'Columbia St', 'Bronxdale', 'Douglaston', 'Pelham Gardens',
       'Olinville', 'Fieldston', 'Highbridge', 'Richmondtown', 'Woodlawn',
       'Bay Terrace', 'Huguenot', 'Manhattan Beach', 'Silver Lake',
       'Castleton Corners', 'Howland Hook', 'Gerritsen Beach',
       'Graniteville', 'Riverdale', 'Great Kills', 'Sea Gate',
       'Grymes Hill', 'Soundview', 'Breezy Point', 'Westerleigh',
       'Bay Terrace, Staten Island', 'Little Neck', 'Todt Hill',
       'Neponsit', 'Willowbrook', 'Chelsea, Staten Island']

class predictForm(FlaskForm):
    print('predictform')
    minimum_nights = StringField('Minimum Nights')
    number_of_reviews = StringField('Number of Review')
    calculated_host_listings_count = StringField('Number of Host Listings')
    availability_365 = StringField('Availability during the Year')
    neighbourhood_group = SelectField(label='Neighbourhood Group', choices = neighbourhood_groups)
    room_type = SelectField(label='Room Type', choices = room_types)
    neighbourhood = SelectField(label='Neighbourhood', choices = neighbourhoods)
    submit = SubmitField("Predict")

@app.route('/', methods = ["GET", "POST"])
def index():
    form = predictForm()
    if form.validate_on_submit():
        session['minimum_nights'] = form.minimum_nights.data
        session['number_of_reviews'] = form.number_of_reviews.data
        session['calculated_host_listings_count'] = form.calculated_host_listings_count.data
        session['availability_365'] = form.availability_365.data
        session['neighbourhood_group'] = form.neighbourhood_group.data
        session['room_type'] = form.room_type.data
        session['neighbourhood'] = form.neighbourhood.data
        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
    content = {}
    content['minimum_nights'] = int(session['minimum_nights'])
    content['number_of_reviews'] = int(session['number_of_reviews'])
    content['calculated_host_listings_count'] = int(session['calculated_host_listings_count'])
    content['availability_365'] = int(session['availability_365'])
    content['neighbourhood_group'] = session['neighbourhood_group']
    content['room_type'] = session['room_type']
    content['neighbourhood'] = session['neighbourhood']
    newPred = pandas.DataFrame([content])
    results = "$" + str(round(return_prediction(model, newPred),2))
    return render_template('prediction.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)