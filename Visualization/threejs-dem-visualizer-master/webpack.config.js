const path = require("path");
const webpack = require("webpack");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CleanWebpackPlugin = require("clean-webpack-plugin");
// const BundleAnalyzerPlugin = require("webpack-bundle-analyzer")
// const CompressionPlugin = require("compression-webpack-plugin");

module.exports = {
  target: "web",

  entry: {
    home: ["./src/js/index.js", "./src/sass/home.sass"],
  },

  output: {
    path: path.join(__dirname, "build"),
    filename: "[chunkhash].js",
    chunkFilename: "[id].bundle.js"
  },

  devtool: "cheap-source-map",

  module: {
    rules: [
      // rule for .js/.jsx files
      {
        test: /\.(js|jsx)$/,
        include: [path.join(__dirname, "js", "src")],
        exclude: [path.join(__dirname, "node_modules")],
        use: {
          loader: "babel-loader"
        }
      },
      // rule for .css/.sass/.scss files
      {
        test: /\.(css|sass|scss)$/,
        use: [
          MiniCssExtractPlugin.loader,
          {
            loader: "css-loader",
            options: {
            //   modules: true,
              // importLoaders allows to configure how many loaders before css-loader should be applied to @imported resources.
              // 0 => no loaders (default); 1 => postcss-loader; 2 => postcss-loader, sass-loader
              importLoaders: 2,
              sourceMap: true,
            }
          },
          {
            loader: "sass-loader",
            options: {
              sourceMap: true
            }
          }
        ]
      },
      // rule for textures (images)
      {
        test: /\.(jpe?g|png|tif)$/i,
        include: path.join(__dirname, "src", "textures"),
        loaders: [
          "file-loader",
          {
            loader: "image-webpack-loader",
            query: {
              progressive: true,
              optimizationLevel: 7,
              interlaced: false,
              pngquant: {
                quality: "65-90",
                speed: 4
              }
            }
          }
        ]
      }
    ]
  },

  optimization: {
    splitChunks: {
      cacheGroups: {
        js: {
          test: /\.js$/,
          name: "commons",
          chunks: "all",
          minChunks: 7
        },
        css: {
          test: /\.(css|sass|scss)$/,
          name: "commons",
          chunks: "all",
          minChunks: 2
        }
      }
    }
  },

  plugins: [
    // new BundleAnalyzerPlugin(),
    new CleanWebpackPlugin(["build"], {
      root: __dirname,
      exclude: ["favicon.ico"],
      verbose: true
    }),
    new MiniCssExtractPlugin({
      filename: "[name].css",
      chunkFilename: "[id].bundle.css"
    }),
    new HtmlWebpackPlugin({
      template: path.join(__dirname, "src", "templates", "index.html"),
      hash: true,
      filename: "index.html",
      chunks: ["commons", "home"]
    }),
    // new CompressionPlugin({
    //   asset: '[path].gz[query]',
    //   algorithm: 'gzip',
    //   test: /\.(js|html)$/, 
    //   threshold: 10240,
    //   minRatio: 0.8,
    // }),
  ],

  devServer: {
    host: "localhost",
    port: 8080,
    contentBase: path.join(__dirname, "build"),
    inline: true, // live reloading
    stats: {
      colors: true,
      reasons: true,
      chunks: false,
      modules: false
    }
  },

  performance: {
    hints: "warning"
  }
};
