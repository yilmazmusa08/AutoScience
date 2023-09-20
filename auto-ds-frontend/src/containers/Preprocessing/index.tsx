import React from "react";
import { InboxOutlined, LoadingOutlined } from "@ant-design/icons";
import { Row, Col, Button, Typography, message, Upload, Space } from "antd";
import { useApp } from "../../context/app.context";
import {
  isFileSizeWithinLimit,
  MAX_FILE_SIZE_MB,
} from "../../utils/validations";
import "./index.css";

const { Dragger } = Upload;

const Preprocessing: React.FC = () => {
  const { preprocess, updateFile, file, loading } = useApp();

  const handleDownload = () => {
    if (file) {
      preprocess({ file });
    }
  };

  return (
    <Col className="preprocessing-container">
      <Row className="preprocessing-title">
        <Typography.Title level={4}>Preprocessing</Typography.Title>
      </Row>
      <Row>
        <Space wrap align="start">
          <Dragger
            style={{ padding: "0 20px" }}
            accept=".xlsx, .xls, .csv"
            maxCount={1}
            onRemove={(file) => {
              updateFile(null);
            }}
            beforeUpload={(file) => {
              if (isFileSizeWithinLimit(file.size)) {
                updateFile(file);
              } else {
                message.error(
                  `File size must be smaller than ${MAX_FILE_SIZE_MB}MB!`,
                );
              }
              return false;
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              Click or drag file to this area to upload
            </p>
            <p className="ant-upload-hint">
              Support for a single upload. (CSV, XLSX, XLS) <br />
              Max file size: {MAX_FILE_SIZE_MB} MB
            </p>
          </Dragger>
          <Button
            type="primary"
            onClick={handleDownload}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <LoadingOutlined /> Preprocessing...
              </>
            ) : (
              "Download"
            )}
          </Button>
        </Space>
      </Row>
    </Col>
  );
};

export default Preprocessing;
