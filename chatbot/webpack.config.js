const path = require('path');

module.exports = {
  entry: './src/embed/widget.js',
  output: {
    filename: 'book-embedded-rag.js',
    path: path.resolve(__dirname, 'dist'),
    library: 'BookRAG',
    libraryTarget: 'umd',
    globalObject: 'this'
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      }
    ]
  },
  mode: 'production',
  optimization: {
    minimize: true
  }
};