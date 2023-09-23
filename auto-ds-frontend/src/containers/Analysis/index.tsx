import React from "react";
import {
  InboxOutlined,
  LoadingOutlined,
  CopyOutlined,
  DownloadOutlined,
} from "@ant-design/icons";
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
  Tooltip,
} from "antd";
import { useApp } from "../../context/app.context";
import { isFileSizeWithinLimit } from "../../utils/validations";
import { downloadFile } from "../../utils/downloadFile";
import { copyToClipboard } from "../../utils/copyToClipboard";
import constant from "../../constants";
import Papa from "papaparse";
import "./index.css";

const { Dragger } = Upload;

const Analysis: React.FC = () => {
  const {
    analyze,
    analysis,
    updateTargetColumns,
    targetColumns,
    updateTargetColumn,
    targetColumn,
    file,
    updateFile,
    loadingAnalysis,
  } = useApp();

  const handleAnalysis = () => {
    if (file) {
      analyze({ file, target_column: targetColumn });
    }
  };

  return (
    <Col className="analysis-container">
      <Row className="analysis-title">
        <Typography.Title level={4}>Analysis</Typography.Title>
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
                message.error(
                  `File size must be smaller than ${constant.MaxFileSizeMB}MB!`,
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
              Max file size: {constant.MaxFileSizeMB} MB
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
            onClick={handleAnalysis}
            disabled={!file || loadingAnalysis}
          >
            {loadingAnalysis ? (
              <>
                <LoadingOutlined /> Analyzing...
              </>
            ) : (
              "Analyze"
            )}
          </Button>
        </Space>
      </Row>
      <Row justify="space-between">
        <Typography.Title className="results-title" level={5}>
          Results
        </Typography.Title>
        {analysis && (
          <Space wrap>
            <Tooltip title="Copy to Clipboard">
              <Button
                type="primary"
                icon={<CopyOutlined />}
                onClick={() =>
                  copyToClipboard(JSON.stringify(analysis, null, 2))
                }
              />
            </Tooltip>
            <Button
              type="primary"
              icon={<DownloadOutlined />}
              onClick={() => downloadFile(analysis, "results.json")}
            >
              Download
            </Button>
          </Space>
        )}
      </Row>
      <Spin wrapperClassName="results-container" spinning={loadingAnalysis}>
        <Row className="results-container">
          {analysis && (
            <Typography.Paragraph className="results-text">
              <pre className="results-pre">
                {JSON.stringify(analysis, null, 2)}
              </pre>
            </Typography.Paragraph>
          )}
        </Row>
      </Spin>
    </Col>
  );
};

export default Analysis;
