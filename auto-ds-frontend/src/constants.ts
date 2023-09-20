const constants = {
  BaseUrl: process.env.REACT_APP_BASE_API_URL || "http://localhost/api",
  MaxFileSizeMB: process.env.REACT_APP_MAX_FILE_SIZE_MB || 20,
};

export default constants;
