import http from "k6/http";
import {sleep} from "k6";


// QA environment by default (if not in Kubernetes)
export const ACE_URL = __ENV.KUBERNETES_SERVICE_HOST ?
    "http://ace-api.datascience.svc.cluster.local" :
    __ENV.STAGE === "prod" ?
        "https://ace.talabat.com" :
        __ENV.STAGE === "qa" ?
            "https://ace-qa.dhhmena.com" : `http://localhost:${__ENV.PORT}`;

export class AceApiTester {
    constructor(baseUrl = ACE_URL) {
        this.baseUrl = baseUrl;
    }

    rankEndpoint(accessToken, request) {
        let response = http.post(
            this.baseUrl + "/vendor_ranking/sort",
            request,
            {
                headers: {
                    "Content-Type": "application/json",
                    "Authorization":  "Bearer " + accessToken,
                },
            }
        );

        // TODO Check response JSON structure

        // A small pause is usually required, more on https://k6.io/docs/using-k6/test-life-cycle/#the-default-function-life-cycle
        sleep(0.1);
    }
}

export const ACE_ENDPOINT_PATH = __ENV.ACE_ENDPOINT_PATH ? __ENV.ACE_ENDPOINT_PATH : ""

export class UltronApiTester {
    constructor(baseUrl = ACE_URL, endpointPath = ACE_ENDPOINT_PATH) {
        this.baseUrl = baseUrl;
        this.endpointPath = endpointPath;
    }

    rankEndpoint(accessToken, request) {
        let response = http.post(
            this.baseUrl + this.endpointPath,
            request,
            {
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + accessToken,
                },
            }
        );

        // A small pause is usually required, more on https://k6.io/docs/using-k6/test-life-cycle/#the-default-function-life-cycle
        sleep(0.1);
    }
}
