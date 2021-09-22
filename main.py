from RF import RandomForest
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with

app = Flask(__name__)
api = Api(app)

video_put_args = reqparse.RequestParser()
video_put_args.add_argument("title", type=str, help="Title of bug report", required=True)
video_put_args.add_argument("content", type=str, help="Content of bug report", required=True)


resource_fields = {
	'title': fields.Integer,
	'content': fields.String,
}

class BugReport(Resource):
    def get(self):
        return "Bo sung sau"
        
    @marshal_with(resource_fields)
    def post(self):
        args = video_put_args.parse_args()
        print(args)

api.add_resource(BugReport, "/last")

if __name__ == "__main__":
    rfModel = RandomForest.RandomForestModel("Dataset/stock_data.csv")
    rfModel.Fit()
    rfModel.GetAccuracy()
    input = rfModel.nlpModel.ConvertRow("NG - nice PNF BY - breakout - need follow thru  ")
    rfModel.Predict(input)
    app.run(debug=True)
    
