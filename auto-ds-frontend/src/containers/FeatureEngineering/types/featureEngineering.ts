export enum OperationType {
  PerformOperation = "perform_operation",
  TakeFirstN = "take_first_n",
  TakeLastN = "take_last_n",
  CreateTransformedColumn = "create_transformed_column",
  ScaleColumnInDataframe = "scale_column_in_dataframe",
  ReplaceValues = "replace_values",
  CreateFlagColumn = "create_flag_column",
  RemoveOutliers = "remove_outliers",
}

export enum MethodType {
  Add = "add",
  Substract = "substract",
  Multiply = "multiply",
  Divide = "divide",
}

export enum TransformType {
  Log = "log",
  Power = "power",
  Root = "root",
}

export enum ScalerType {
  StandardScaler = "StandardScaler",
  MinMaxScaler = "MinMaxScaler",
  RobustScaler = "RobustScaler",
}

export const operationTypeData = [
  {
    value: OperationType.PerformOperation,
    label: "Perform Operation",
  },
  {
    value: OperationType.TakeFirstN,
    label: "Take First N",
  },
  {
    value: OperationType.TakeLastN,
    label: "Take Last N",
  },
  {
    value: OperationType.CreateTransformedColumn,
    label: "Create Transformed Column",
  },
  {
    value: OperationType.ScaleColumnInDataframe,
    label: "Scale Column In DataFrame",
  },
  {
    value: OperationType.ReplaceValues,
    label: "Replace Values",
  },
  {
    value: OperationType.CreateFlagColumn,
    label: "Create Flag Column",
  },
  {
    value: OperationType.RemoveOutliers,
    label: "Remove Outliers",
  },
];

export const methodTypeData = [
  {
    value: MethodType.Add,
    label: "Add",
  },
  {
    value: MethodType.Substract,
    label: "Substract",
  },
  {
    value: MethodType.Multiply,
    label: "Multiply",
  },
  {
    value: MethodType.Divide,
    label: "Divide",
  },
];

export const transformTypeData = [
  {
    value: TransformType.Log,
    label: "Log",
  },
  {
    value: TransformType.Power,
    label: "Power",
  },
  {
    value: TransformType.Root,
    label: "Root",
  },
];

export const scalerTypeData = [
  {
    value: ScalerType.StandardScaler,
    label: "Standard Scaler",
  },
  {
    value: ScalerType.MinMaxScaler,
    label: "Min Max Scaler",
  },
  {
    value: ScalerType.RobustScaler,
    label: "Robust Scaler",
  },
];

export interface IFeatureEngineeringRequest {
  file: File;
  operation: OperationType;
  col_name?: string | null;
  col1?: string | null;
  col2?: string | null;
  method: MethodType;
  n?: number | null;
  Q1?: number | null;
  Q2?: number | null;
  remove: boolean;
  transform_type?: TransformType | null;
  scaler_type?: ScalerType | null;
  val_search?: string | null;
  val_replace?: string | null;
}
