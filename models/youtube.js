const mongoose = require('mongoose');
const { Schema } = mongoose;

// 引入mongodb
const videoSchema = new Schema({
  title:  String, 
  url: String, 
  transcript: String,
  summary: String
});

const Video = mongoose.model('Video', videoSchema);

mongoose.connect('mongodb://localhost:27017/youtube', {useNewUrlParser: true, useUnifiedTopology: true});


