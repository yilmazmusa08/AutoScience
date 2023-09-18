import React from "react";
import { InboxOutlined, LoadingOutlined } from "@ant-design/icons";
import {
  Row,
  Col,
  Button,
  Typography,
  message,
  Upload,
  Space,
  Select,
  Spin,
} from "antd";
import { useApp } from "../../context/app.context";
import Papa from "papaparse";
import "./index.css";

const { Dragger } = Upload;

const Models: React.FC = () => {
  const {
    runModels,
    models,
    updateTargetColumns,
    targetColumns,
    updateTargetColumn,
    targetColumn,
    file,
    updateFile,
    loading,
  } = useApp();

  const handleModels = () => {
    if (file) {
      runModels({ file, target_column: targetColumn });
    }
  };

  return (
    <Col className="models-container">
      <Row className="models-title">
        <Typography.Title level={4}>Models</Typography.Title>
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
                const reader = new FileReader();

                reader.onload = ({ target }) => {
                  const data = target?.result?.toString();
                  if (data) {
                    Papa.parse<string[]>(data, {
                      complete: ({ data }) => {
                        updateTargetColumns(data[0]);
                      },
                    });
                  }
                };
                reader.readAsBinaryString(file);
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
          <Select
            allowClear
            placeholder="Select the target column"
            disabled={!file}
            onSelect={(value: string) => updateTargetColumn(value)}
            onClear={() => updateTargetColumn(null)}
            value={targetColumn}
            options={targetColumns?.map((column) => ({
              value: column,
              label: column,
            }))}
          />
          <Button
            type="primary"
            onClick={handleModels}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <LoadingOutlined /> Running Models...
              </>
            ) : (
              "Run Models"
            )}
          </Button>
        </Space>
      </Row>
      <Row>
        <Typography.Title level={5}>Results</Typography.Title>
      </Row>
      <Spin wrapperClassName="results-container" spinning={loading}>
        <Row className="results-container">
          {models && (
            <Typography.Paragraph className="results-text">
              <pre className="results-pre">
                {JSON.stringify(models, null, 2)}
              </pre>
            </Typography.Paragraph>
          )}
        </Row>
      </Spin>
    </Col>
  );
};

export default Models;
