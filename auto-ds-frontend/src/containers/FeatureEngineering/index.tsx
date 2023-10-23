import React, { useState } from "react";
import { InboxOutlined, LoadingOutlined } from "@ant-design/icons";
import {
  Row,
  Col,
  Button,
  Typography,
  Form,
  message,
  Upload,
  Select,
  Space,
  InputNumber,
  Checkbox,
  Input,
} from "antd";
import { useApp } from "../../context/app.context";
import { isFileSizeWithinLimit } from "../../utils/validations";
import constant from "../../constants";
import {
  OperationType,
  operationTypeData,
  MethodType,
  methodTypeData,
  TransformType,
  transformTypeData,
  ScalerType,
  scalerTypeData,
} from "./types/featureEngineering";
import Papa from "papaparse";
import "./index.css";

const { Dragger } = Upload;

const FeatureEngineering: React.FC = () => {
  const {
    runFeatureEngineering,
    updateFile,
    file,
    updateTargetColumns,
    targetColumns,
    loadingFeatureEngineering,
  } = useApp();
  const [form] = Form.useForm();

  const onFinish = (formData: any) => {
    if (file) {
      runFeatureEngineering({ file, ...formData });
    }
  };

  return (
    <Col className="feature-engineering-container">
      <Row className="feature-engineering-title">
        <Typography.Title level={4}>Feature Engineering</Typography.Title>
      </Row>
      <Row gutter={16}>
        <Col span={6}>
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
        </Col>
        <Col span={18}>
          <Form
            layout="vertical"
            form={form}
            initialValues={{
              operation: operationTypeData[0].value,
              method: methodTypeData[0].value,
              scaler: null,
              col_name: null,
              col1: null,
              col2: null,
              n: null,
              Q1: null,
              Q2: null,
              remove: false,
              val_search: "",
              val_replace: "",
            }}
            onFinish={onFinish}
          >
            <Space direction="vertical" wrap>
              <Row gutter={8}>
                <Col span={6}>
                  <Form.Item
                    label="Operation"
                    name="operation"
                    rules={[
                      {
                        required: true,
                        message: "Please select an operation!",
                      },
                    ]}
                  >
                    <Select
                      placeholder="Select Operation"
                      disabled={!file}
                      options={operationTypeData}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item label="Column Name" name="col_name">
                    <Select
                      allowClear
                      placeholder="Select column name"
                      disabled={!file}
                      options={targetColumns?.map((column) => ({
                        value: column,
                        label: column,
                      }))}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item label="Col 1" name="col1">
                    <Select
                      allowClear
                      placeholder="Select Col 1"
                      disabled={!file}
                      options={targetColumns?.map((column) => ({
                        value: column,
                        label: column,
                      }))}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item label="Col 2" name="col2">
                    <Select
                      allowClear
                      placeholder="Select Col 2"
                      disabled={!file}
                      options={targetColumns?.map((column) => ({
                        value: column,
                        label: column,
                      }))}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
              </Row>
              <Row align="middle" gutter={8}>
                <Col span={6}>
                  <Form.Item
                    label="Method"
                    name="method"
                    rules={[
                      {
                        required: true,
                        message: "Please select a method!",
                      },
                    ]}
                  >
                    <Select
                      placeholder="Select Method"
                      disabled={!file}
                      options={methodTypeData}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item label="n" name="n">
                    <InputNumber placeholder="n" style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item label="Q1" name="Q1">
                    <InputNumber placeholder="Q1" style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col span={4}>
                  <Form.Item label="Q2" name="Q2">
                    <InputNumber placeholder="Q2" style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item label="Remove" name="remove">
                    <Checkbox>Remove</Checkbox>
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={8}>
                <Col span={6}>
                  <Form.Item label="Transform Type" name="transform">
                    <Select
                      allowClear
                      placeholder="Select Transform Type"
                      disabled={!file}
                      options={transformTypeData}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item label="Scaler Type" name="scaler">
                    <Select
                      allowClear
                      placeholder="Select Scaler Type"
                      disabled={!file}
                      options={scalerTypeData}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item label="Value Search" name="val_search">
                    <Input placeholder="Value Search" />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item label="Value Replace" name="val_replace">
                    <Input placeholder="Value Replace" />
                  </Form.Item>
                </Col>
              </Row>
              <Row justify="end">
                <Button
                  type="primary"
                  htmlType="submit"
                  disabled={!file || loadingFeatureEngineering}
                >
                  {loadingFeatureEngineering ? (
                    <>
                      <LoadingOutlined /> Feature Engineering...
                    </>
                  ) : (
                    "Run Feature Engineering"
                  )}
                </Button>
              </Row>
            </Space>
          </Form>
        </Col>
      </Row>
    </Col>
  );
};

export default FeatureEngineering;
