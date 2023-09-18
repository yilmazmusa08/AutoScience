import React from "react";
import { InboxOutlined } from "@ant-design/icons";
import {
  Row,
  Col,
  Button,
  Typography,
  message,
  Upload,
  Space,
} from "antd";
import { useApp } from "../../context/app.context";
import { preprocessRequest } from "../../services/requests/preprocess";
import "./index.css";

const { Dragger } = Upload;

const Preprocessing: React.FC = () => {
  const { updateFile, file } = useApp();

  const handleDownload = () => {
    if (file) {
      preprocessRequest({ file });
    }
  };

  return (
    <Col>
      <Row>
        <Typography.Title level={4} style={{ marginBottom: 20 }}>
          Preprocessing
        </Typography.Title>
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
              const isLt20M = file.size / 1024 / 1024 < 20;
              if (isLt20M) {
                updateFile(file);
              } else {
                message.error("File must smaller than 20MB!");
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
              Max file size: 20 MB
            </p>
          </Dragger>
          <Button
            type="primary"
            onClick={handleDownload}
            disabled={!file}
            // loading={uploading}
          >
            Download
          </Button>
        </Space>
      </Row>
    </Col>
  );
};

export default Preprocessing;
