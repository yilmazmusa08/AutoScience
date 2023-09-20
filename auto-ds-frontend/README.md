To run the frontend standalone:

### Prerequisites

Before you begin, ensure you have met the following requirements:

- [Node.js](https://nodejs.org/) installed
- [npm](https://www.npmjs.com/) or [Yarn](https://yarnpkg.com/) installed


### Installation

Install dependencies:

```shell
npm install
```

### Configuration

Put the copy of the `.env.dev` file into a new `.env` file. 

```shell
REACT_APP_BASE_API_URL=http://localhost/api
REACT_APP_MAX_FILE_SIZE_MB=20
```

### Run

```shell
npm start
```

### Running Tests

```shell
npm test
```

### Linting

```shell
npm run lint
```