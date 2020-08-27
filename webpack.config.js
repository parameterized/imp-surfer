
const path = require('path');

module.exports = function (env, argv) {
    let prod = env && env.production;
    return {
        mode: prod ? 'production' : 'development',
        entry: `./src/index.js`,
        output: {
            path: path.join(__dirname, 'public'),
            filename: `bundle.js`
        },
        module: {
            rules: [
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    loader: 'babel-loader'
                }
            ]
        },
        devtool: 'eval-cheap-source-map',
        devServer: {
            contentBase: path.join(__dirname, 'public'),
            compress: true,
            port: 8080
        }
    };
};
