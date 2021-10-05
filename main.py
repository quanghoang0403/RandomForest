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
class GetAll(Resource):
    def get(self):
        return "Bo sung sau"

class Create(Resource):
    def get(self):
        args = video_put_args.parse_args()
        rfModel = RandomForest.RandomForestModel("Dataset/dataset2.csv")
        rfModel.Fit()
        response = rfModel.WriteBug(args.title, args.content)
        jsonList = []
        for index, row in response.iterrows():
            jsonList.append({"id_sentence": row["id_sentence"], 
            "title":row["title"],
            "id_bug":row["id_bug"],
            "id_comment":row["id_comment"],
            "position":row["position"],
            "content":row["content"],
            "preP":row["preP"],
            "result":row["result"]
            })
        return(jsonify(jsonList))

api.add_resource(GetAll, "/getall")
api.add_resource(Create, "/create")
if __name__ == "__main__":
    app.run(debug=True)
    
