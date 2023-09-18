import {
  createContext,
  useCallback,
  useContext,
  useState,
  useEffect,
} from "react";
import { IPreprocessingRequest } from "../containers/Preprocessing/types/preprocessing";
import {
  IAnalysis,
  IAnalysisRequest,
} from "../containers/Analysis/types/analysis";
import { IModels, IModelsRequest } from "../containers/Models/types/models";
import {
  ILogin,
  IRegister,
  IAuthResponse,
} from "../containers/Authentication/types/authentication";

import { preprocessRequest } from "../services/requests/preprocess";
import { analyzeRequest } from "../services/requests/analyze";
import { modelsRequest } from "../services/requests/models";
import { loginRequest } from "../services/requests/login";
import { registerRequest } from "../services/requests/register";

import { UserStorage, AuthUser } from "../services/UserStorage";

interface AppContextState {
  analysis: IAnalysis | null;
  models: IModels | null;
  authUser: IAuthResponse | null;
  targetColumns: string[] | undefined;
  targetColumn: string | null;
  file: File | null;
  preprocess: (payload: IPreprocessingRequest) => void;
  analyze: (payload: IAnalysisRequest) => void;
  runModels: (payload: IModelsRequest) => void;
  login: (payload: ILogin) => void;
  logout: () => void;
  register: (payload: IRegister) => void;
  updateTargetColumns: (targetColumns?: string[]) => void;
  updateTargetColumn: (targetColumn: string | null) => void;
  updateFile: (file: File | null) => void;
  loading: boolean;
  updateLoading: (value: boolean) => void;
}

const defaultAppContext: AppContextState = {
  analysis: null,
  models: null,
  authUser: null,
  targetColumns: [],
  targetColumn: null,
  file: null,
  preprocess: () => {},
  analyze: () => {},
  runModels: () => {},
  login: () => {},
  logout: () => {},
  register: () => {},
  updateTargetColumns: () => {},
  updateTargetColumn: () => {},
  updateFile: () => {},
  loading: false,
  updateLoading: () => {},
};

const useAppContext = (props: AppContextState): AppContextState => {
  const [authUser, setAuthUser] = useState(props.authUser);
  const [analysis, setAnalysis] = useState(props.analysis);
  const [models, setModels] = useState(props.models);
  const [targetColumns, setTargetColumns] = useState(props.targetColumns);
  const [targetColumn, setTargetColumn] = useState(props.targetColumn);
  const [file, setFile] = useState(props.file);
  const [loading, setLoading] = useState(props.loading);

  const updateAuthUser = useCallback(
    (authUser: AppContextState["authUser"]) => {
      setAuthUser(authUser);
    },
    [],
  );

  const updateAnalysis = useCallback(
    (analysis: AppContextState["analysis"]) => {
      setAnalysis(analysis);
    },
    [],
  );

  const updateModels = useCallback((models: AppContextState["models"]) => {
    setModels(models);
  }, []);

  const updateTargetColumns = useCallback(
    (targetColumns?: AppContextState["targetColumns"]) => {
      setTargetColumns(targetColumns);
    },
    [],
  );

  const updateTargetColumn = useCallback(
    (targetColumn: AppContextState["targetColumn"]) => {
      setTargetColumn(targetColumn);
    },
    [],
  );

  const updateFile = useCallback((file: AppContextState["file"]) => {
    setFile(file);
    updateTargetColumns(defaultAppContext.targetColumns);
    updateTargetColumn(defaultAppContext.targetColumn);
    updateAnalysis(defaultAppContext.analysis);
    updateModels(defaultAppContext.models);
  }, []);

  const updateLoading = useCallback((loading: AppContextState["loading"]) => {
    setLoading(loading);
  }, []);

  const preprocess = (payload: IPreprocessingRequest) => {
    updateLoading(true);
    preprocessRequest(payload)
      .then((data) => {
        const url = window.URL.createObjectURL(new Blob([data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", payload.file.name);
        document.body.appendChild(link);
        link.click();
        link?.parentNode?.removeChild(link);
        updateLoading(false);
      })
      .catch((error: any) => {
        updateLoading(false);
        throw error;
      });
  };

  const analyze = (payload: IAnalysisRequest) => {
    updateLoading(true);
    analyzeRequest(payload)
      .then((data: IAnalysis) => {
        updateAnalysis(data);
        updateLoading(false);
      })
      .catch((error: any) => {
        updateLoading(false);
        throw error;
      });
  };

  const runModels = (payload: IModelsRequest) => {
    updateLoading(true);
    modelsRequest(payload)
      .then((data: IModels) => {
        updateModels(data);
        updateLoading(false);
      })
      .catch((error: any) => {
        updateLoading(false);
        throw error;
      });
  };

  const login = (payload: ILogin) => {
    updateLoading(true);
    loginRequest(payload)
      .then((data: IAuthResponse) => {
        UserStorage.storeUser(data);
        updateAuthUser(data);
        updateLoading(false);
      })
      .catch((error: any) => {
        updateLoading(false);
        throw error;
      });
  };

  const register = (payload: IRegister) => {
    updateLoading(true);
    registerRequest(payload)
      .then((data: IAuthResponse) => {
        UserStorage.storeUser(data);
        updateLoading(false);
        updateAuthUser(data);
      })
      .catch((error: any) => {
        updateLoading(false);
        throw error;
      });
  };

  const logout = () => {
    UserStorage.clear();
    updateAuthUser(defaultAppContext.authUser);
    updateFile(defaultAppContext.file);
  };

  useEffect(() => {
    if (UserStorage.isAuthenticated()) {
      const id = UserStorage.getUser(AuthUser.ID);
      const username = UserStorage.getUser(AuthUser.Username);
      const email = UserStorage.getUser(AuthUser.Email);
      const accessToken = UserStorage.getUser(AuthUser.AccessToken);
      const refreshToken = UserStorage.getUser(AuthUser.RefreshToken);
      if (id && username && email && accessToken) {
        updateAuthUser({
          access: accessToken,
          refresh: refreshToken || "",
          user: { pk: parseInt(id, 10), username, email },
        });
      }
    }
  }, []);

  return {
    analysis,
    models,
    preprocess,
    analyze,
    runModels,
    login,
    logout,
    register,
    authUser,
    targetColumns,
    targetColumn,
    updateTargetColumns,
    updateTargetColumn,
    file,
    updateFile,
    loading,
    updateLoading,
  };
};

const AppContext = createContext<AppContextState>(defaultAppContext);

export const useApp = (): AppContextState => {
  return useContext(AppContext);
};

export const AppContextConsumer = AppContext.Consumer;

export const AppContextProvider: React.FC<{ children: React.ReactElement }> = ({
  children,
}) => {
  const {
    analysis,
    models,
    preprocess,
    analyze,
    runModels,
    login,
    logout,
    register,
    authUser,
    targetColumns,
    targetColumn,
    updateTargetColumns,
    updateTargetColumn,
    file,
    updateFile,
    loading,
    updateLoading,
  } = useAppContext(defaultAppContext);

  return (
    <AppContext.Provider
      value={{
        analysis,
        models,
        preprocess,
        analyze,
        runModels,
        login,
        logout,
        register,
        authUser,
        targetColumns,
        targetColumn,
        updateTargetColumns,
        updateTargetColumn,
        file,
        updateFile,
        loading,
        updateLoading,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
